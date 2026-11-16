"""Distributed correctness suite (spec section 5), shared by pytest (gloo/CPU) and torchrun (NCCL/GPU).

For every case, deterministic global inputs/weights are generated on CPU, rounded to the case dtype,
and their exact shards are given to TP and TSP. Rank 0 compares both methods against a float64
unsharded reference. TSP outputs are gathered and un-zigzagged here, outside any timed region.

Usage (GPU, D=2):  torchrun --nproc_per_node=2 -m tspfold.correctness --suite gpu --out results/correctness/gpu.jsonl
Usage (D=1 smoke): python -m tspfold.correctness --suite smoke --out results/correctness/smoke_d1.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import ATTN_BACKENDS, BlockShape, ConfigError, parse_dtype
from .dist_utils import all_gather_stacked, init_from_env, precision_settings, set_fp32_strict
from .layout import zigzag_shard, zigzag_unshard
from .metrics import TOLERANCES, compare, max_abs_diff
from .ops import causal_attention, sdpa_backend_context, swiglu_shard
from .reference import reference_attention, reference_mlp
from .tp import TPAttention, TPMLP
from .tsp import TSPAttention, TSPMLP, ring_exchange
from .weights import (
    make_attention_weights,
    make_input,
    make_mlp_weights,
    shard_attention,
    shard_mlp,
)


@dataclass(frozen=True)
class Case:
    block: str
    batch: int
    seq_len: int
    hidden: int
    n_heads: int
    intermediate: int
    dtype: str = "fp32"
    head_bucket: int | None = None
    overlap: bool = True
    seed: int = 0
    backend: str = "sdpa"
    perturb: bool = False
    repeats: int = 2

    @property
    def case_id(self) -> str:
        dims = f"B{self.batch}_S{self.seq_len}_h{self.hidden}"
        dims += f"_H{self.n_heads}" if self.block == "attn" else f"_I{self.intermediate}"
        extra = f"_hb{self.head_bucket or 'all'}_{self.backend}" if self.block == "attn" else ""
        return f"{self.block}_{dims}_{self.dtype}{extra}_ov{int(self.overlap)}_seed{self.seed}"


def _attn_cases(shapes, dtypes, buckets_for, overlaps, seeds, backends=("sdpa",)) -> list[Case]:
    cases = []
    for B, S, h, H in shapes:
        for dtype in dtypes:
            for hb in buckets_for(H):
                for ov in overlaps:
                    for seed in seeds:
                        for be in backends:
                            cases.append(Case("attn", B, S, h, H, 4 * h, dtype, hb, ov, seed, be, perturb=seed == seeds[0]))
    return cases


def _mlp_cases(shapes, dtypes, overlaps, seeds) -> list[Case]:
    return [
        Case("mlp", B, S, h, 4, I, dtype, None, ov, seed, perturb=seed == seeds[0])
        for B, S, h, I in shapes
        for dtype in dtypes
        for ov in overlaps
        for seed in seeds
    ]


def _buckets(world_size: int, extra: tuple[int, ...] = (1,)):
    def fn(n_heads: int) -> list[int | None]:
        hs = n_heads // world_size
        return [None] + [b for b in extra if b < hs and hs % b == 0]

    return fn


def build_suite(name: str, world_size: int) -> list[Case]:
    """Named case grids. Shapes must satisfy the BlockShape contracts for the given world size."""
    D = world_size
    if name == "smoke":
        return [
            Case("attn", 1, 8 * D, 32, 4, 128, "fp32", None, True, 0, perturb=True),
            Case("attn", 2, 8 * D, 32, 4, 128, "fp32", 1, False, 1),
            Case("mlp", 1, 8 * D, 32, 4, 64, "fp32", None, True, 0, perturb=True),
            Case("mlp", 2, 8 * D, 32, 4, 64, "fp32", None, False, 1),
        ]
    if name == "cpu":
        attn_shapes = [(1, 8, 32, 4), (2, 16, 32, 4), (1, 24, 48, 6), (2, 32, 64, 8)]
        mlp_shapes = [(1, 8, 32, 64), (2, 16, 32, 96), (1, 24, 48, 128)]
        cases = _attn_cases(attn_shapes, ["fp32"], _buckets(D, (1, 2)), [True, False], [0, 1])
        cases += _attn_cases([(2, 16, 32, 4)], ["fp32"], _buckets(D), [True], [2], backends=("math", "naive"))
        cases += _attn_cases([(2, 16, 32, 4), (1, 32, 64, 8)], ["bf16"], _buckets(D), [True], [0])
        cases += _mlp_cases(mlp_shapes, ["fp32"], [True, False], [0, 1])
        cases += _mlp_cases([(2, 16, 32, 96)], ["bf16"], [True], [0])
        return cases
    if name == "gpu":
        attn_fp32 = [(1, 64, 256, 4), (2, 256, 256, 8), (1, 1024, 512, 8), (2, 512, 1024, 16)]
        attn_bf16 = [(1, 256, 512, 4), (2, 1024, 1024, 8), (1, 2048, 2048, 16)]
        mlp_fp32 = [(1, 64, 256, 512), (2, 256, 512, 2048), (1, 1024, 1024, 4096)]
        mlp_bf16 = [(1, 256, 512, 2048), (2, 1024, 2048, 8192)]
        cases = _attn_cases(attn_fp32, ["fp32"], _buckets(D, (1, 2)), [True, False], [0, 1])
        cases += _attn_cases(attn_bf16, ["bf16"], _buckets(D, (2,)), [True, False], [0, 1])
        cases += _attn_cases(attn_bf16[:2], ["bf16"], _buckets(D), [True], [0], backends=("flash", "efficient"))
        cases += _mlp_cases(mlp_fp32, ["fp32"], [True, False], [0, 1])
        cases += _mlp_cases(mlp_bf16, ["bf16"], [True, False], [0, 1])
        return cases
    raise ValueError(f"unknown suite {name!r}")


def _gather_zigzag(y_local: torch.Tensor, group) -> torch.Tensor:
    """Correctness-only: all-gather TSP local outputs and undo the zigzag layout."""
    gathered = all_gather_stacked(y_local.contiguous(), group)
    return zigzag_unshard(list(gathered.unbind(0)), dim=1)


def _perturb_positions(seq_len: int, chunk: int) -> list[int]:
    raw = [chunk, 2 * chunk - 1, seq_len - chunk + 1]
    return sorted({min(max(t, 1), seq_len - 1) for t in raw})


def _unsharded(case: Case, x: torch.Tensor, w, device: torch.device) -> torch.Tensor:
    """Single-device computation with the same kernels/precision as the methods (error context only)."""
    with torch.inference_mode():
        x = x.to(device)
        if case.block == "attn":
            w = w.to(device)
            B, S, h, H = case.batch, case.seq_len, case.hidden, case.n_heads
            q, k, v = (torch.nn.functional.linear(x, wm).view(B, S, H, h // H) for wm in (w.wq, w.wk, w.wv))
            with sdpa_backend_context(case.backend):
                o = causal_attention(q, k, v, case.backend)
            return o.reshape(B, S, h) @ w.wo.t()
        packed = shard_mlp(w, 0, 1).to(device)
        return swiglu_shard(x.reshape(-1, case.hidden), packed).view(x.shape)


def run_case(case: Case, device: torch.device, group=None) -> list[dict]:
    """Run TP and TSP for one case on every rank; rank 0 returns result records (others return [])."""
    rank, D = dist.get_rank(group), dist.get_world_size(group)
    dtype = parse_dtype(case.dtype)
    shape = BlockShape(case.batch, case.seq_len, case.hidden, case.n_heads, case.intermediate, D)
    atol, rtol = TOLERANCES[case.dtype]
    x = make_input(case.batch, case.seq_len, case.hidden, case.seed, dtype)

    if case.block == "attn":
        w = make_attention_weights(case.hidden, case.seed, dtype)
        shard = shard_attention(w, rank, D, case.n_heads).to(device)
        methods = {
            "tp": TPAttention(shard, shape, group=group, backend=case.backend),
            "tsp": TSPAttention(
                shard, shape, group=group, head_bucket=case.head_bucket, overlap=case.overlap, backend=case.backend
            ),
        }
        reference = lambda xx: reference_attention(xx, w, case.n_heads)  # noqa: E731
    elif case.block == "mlp":
        w = make_mlp_weights(case.hidden, case.intermediate, case.seed, dtype)
        shard = shard_mlp(w, rank, D).to(device)
        methods = {
            "tp": TPMLP(shard, shape, group=group),
            "tsp": TSPMLP(shard, shape, group=group, overlap=case.overlap),
        }
        reference = lambda xx: reference_mlp(xx, w)  # noqa: E731
    else:
        raise ConfigError(f"unknown block {case.block!r}")

    def run(method: str, xg: torch.Tensor) -> torch.Tensor:
        if method == "tp":
            return methods["tp"](xg.to(device))
        return _gather_zigzag(methods["tsp"](zigzag_shard(xg, rank, D).to(device)), group)

    y_ref = reference(x) if rank == 0 else None
    records = []
    if rank == 0:
        y_uns = _unsharded(case, x, w, device)
        records.append(_record(case, D, "unsharded", compare(y_uns, y_ref, atol, rtol), {}))

    for method in ("tp", "tsp"):
        t0 = time.perf_counter()
        y = run(method, x)
        extra: dict = {"repeat_max_diff": 0.0}
        for _ in range(case.repeats):
            extra["repeat_max_diff"] = max(extra["repeat_max_diff"], max_abs_diff(run(method, x), y))
        if method == "tp":  # TP output must be identical on every rank after the all-reduce
            all_y = all_gather_stacked(y.contiguous(), group)
            extra["cross_rank_max_diff"] = max(max_abs_diff(all_y[r], all_y[0]) for r in range(D))
        if case.perturb:
            noise = make_input(case.batch, case.seq_len, case.hidden, case.seed + 7919, dtype)
            checks = []
            for t in _perturb_positions(case.seq_len, shape.chunk_len):
                xp = x.clone()
                xp[:, t:] = noise[:, t:]
                yp = run(method, xp)
                checks.append(
                    {
                        "t": t,
                        "prefix_max_diff": max_abs_diff(yp[:, :t], y[:, :t]),
                        "suffix_changed": max_abs_diff(yp[:, t:], y[:, t:]) > 0,
                    }
                )
            extra["perturb"] = checks
        extra["wall_s"] = time.perf_counter() - t0
        if rank == 0:
            records.append(_record(case, D, method, compare(y, y_ref, atol, rtol), extra))
    return records


def _record(case: Case, world_size: int, method: str, metrics: dict, extra: dict) -> dict:
    rec = {"case_id": case.case_id, "method": method, "world_size": world_size, **dataclasses.asdict(case)}
    rec.update(metrics)
    rec.update(extra)
    problems = []
    if not metrics["finite"]:
        problems.append("non-finite output")
    if not metrics["allclose"]:
        problems.append("allclose failed")
    if extra.get("repeat_max_diff", 0.0) > metrics["atol"]:
        problems.append("repeated calls inconsistent")
    if extra.get("cross_rank_max_diff", 0.0) > 0.0:
        problems.append("TP output differs across ranks")
    for chk in extra.get("perturb", []):
        if chk["prefix_max_diff"] != 0.0:
            problems.append(f"future perturbation at t={chk['t']} changed earlier outputs")
        if not chk["suffix_changed"]:
            problems.append(f"perturbation at t={chk['t']} had no effect")
    rec["problems"] = problems
    rec["passed"] = not problems if method != "unsharded" else metrics["finite"]
    return rec


def ring_check(device: torch.device, group=None) -> dict:
    """Ring peer exchange: rank p must receive exactly rank p-1's buffer (and repeated use is stable)."""
    rank, D = dist.get_rank(group), dist.get_world_size(group)
    send = torch.full((4, 8), float(rank), device=device)
    ok = True
    for _ in range(3):
        recv = torch.empty_like(send)
        for req in ring_exchange(send, recv, group):
            req.wait()
        ok &= bool((recv == float((rank - 1) % D)).all().item()) and bool((send == float(rank)).all().item())
    flag = torch.tensor([int(ok)], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return {"case_id": "ring_exchange", "method": "tsp", "world_size": D, "passed": bool(flag.item()), "problems": []}


def run_suite(cases: list[Case], device: torch.device, group=None, log=None) -> list[dict]:
    records = []
    if dist.get_world_size(group) > 1:
        records.append(ring_check(device, group))
    for i, case in enumerate(cases):
        recs = run_case(case, device, group)
        records.extend(recs)
        if log is not None and recs:
            status = "ok" if all(r["passed"] for r in recs) else "FAIL"
            errs = " ".join(f"{r['method']}={r['max_abs_err']:.2e}" for r in recs)
            log(f"[{i + 1}/{len(cases)}] {status} {case.case_id} max_abs_err {errs}")
    return records


def environment_summary(device: torch.device) -> dict:
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "precision": precision_settings(),
    }
    if device.type == "cuda":
        info.update(
            gpu=torch.cuda.get_device_name(device),
            cuda=torch.version.cuda,
            nccl=".".join(map(str, torch.cuda.nccl.version())),
        )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="smoke", choices=["smoke", "cpu", "gpu"])
    ap.add_argument("--out", required=True, help="JSONL output path (written by rank 0)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default=None, help="only run cases of this dtype")
    ap.add_argument("--block", default=None, choices=["attn", "mlp"], help="only run this block")
    ap.add_argument(
        "--attn-backend", choices=ATTN_BACKENDS,
        help="override default attention cases; omit the suite's separate forced-backend cases",
    )
    args = ap.parse_args(argv)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    rank, world, device = init_from_env(args.device)
    set_fp32_strict(True)  # TF32 off: FP32 diagnostics must not be silently downgraded
    cases = [
        dataclasses.replace(c, backend=args.attn_backend)
        if args.attn_backend is not None and c.block == "attn" else c
        for c in build_suite(args.suite, world)
        if (args.dtype is None or c.dtype == args.dtype) and (args.block is None or c.block == args.block)
        and (args.attn_backend is None or c.block != "attn" or c.backend == "sdpa")
    ]
    log = print if rank == 0 else None
    if log:
        log(f"suite={args.suite} D={world} device={device} cases={len(cases)}")
    records = run_suite(cases, device, log=log)
    ok = True
    if rank == 0:
        env = environment_summary(device)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps({**rec, "env": env}) + "\n")
        failed = [r for r in records if not r["passed"]]
        ok = not failed
        for r in failed:
            log(f"FAILED {r['case_id']} [{r['method']}]: {r.get('problems')}")
        log(f"{len(records) - len(failed)}/{len(records)} records passed -> {args.out}")
    flag = torch.tensor([int(ok)], device=device)
    dist.broadcast(flag, src=0)
    dist.destroy_process_group()
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
