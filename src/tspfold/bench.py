"""Benchmark ONE method x block x configuration in a fresh process group (spec section 6).

  torchrun --nproc_per_node=2 -m tspfold.bench --method tsp --block attn --batch 1 --seq-len 4096 \
      --out results/raw/main/attn-tsp-B1-S4096.jsonl

Timing: each iteration starts after a barrier (outside the timed window), is bracketed by device-wide
synchronize() (waits for compute and NCCL streams), and is aggregated as the MAX elapsed time across
ranks. Memory: peak allocator stats are reset after warmup and read after the measured iterations.
Only the rank's packed weight shard and its own input live on the GPU; data are generated on CPU.
Failures: each rank writes <status-dir>/<case-id>.rank<r>.status.json; on OOM/error the rank exits
immediately (no further collectives) so torchrun tears down its peer instead of leaving it blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from .config import BLOCKS, METHODS, BlockShape, ConfigError, check_attn_backend, parse_dtype
from .dist_utils import init_from_env, set_fp32_strict, synchronize
from .env import collect_env
from .layout import zigzag_chunk_ids
from .theory import activation_io_bytes, comm_bytes, persistent_bytes
from .tp import TPAttention, TPMLP
from .tsp import TSPAttention, TSPMLP
from .weights import (
    make_attention_weights,
    make_input,
    make_input_range,
    make_mlp_weights,
    shard_attention,
    shard_mlp,
)

SCHEMA = "tspfold.bench/1"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--intermediate", type=int, default=8192)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--head-bucket", type=int, default=0, help="TSP heads per K/V gather bucket; 0 = all local heads")
    ap.add_argument("--overlap", type=int, default=1, choices=[0, 1], help="TSP: prefetch/overlap weight transfers")
    ap.add_argument("--attn-backend", default="sdpa")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trials", type=int, default=1, help="in-process trials (independent trials = separate launches)")
    ap.add_argument("--trial-index", type=int, default=0, help="index of this independent launch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", type=int, default=0, help="profile N extra iterations for a kernel-time breakdown")
    ap.add_argument("--fp32-strict", type=int, default=1, choices=[0, 1], help="disable TF32 (only affects fp32)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--pg-timeout", type=float, default=300.0)
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--out", required=True, help="JSONL file; rank 0 appends one record per in-process trial")
    ap.add_argument("--status-dir", default=None, help="per-rank status files (default: <out dir>/status)")
    return ap


def case_id_for(args) -> str:
    if args.case_id:
        return args.case_id
    attn = f"-{args.attn_backend}" if args.block == "attn" else ""
    tsp = ""
    if args.method == "tsp":
        tsp = (f"-hb{args.head_bucket or 'all'}" if args.block == "attn" else "") + f"-ov{args.overlap}"
    return (
        f"{args.block}-{args.method}-B{args.batch}-S{args.seq_len}-h{args.hidden}-H{args.heads}-I{args.intermediate}"
        f"-{args.dtype}{attn}{tsp}-seed{args.seed}-t{args.trial_index}"
    )


def base_record(args, world: int) -> dict:
    return {
        "schema": SCHEMA,
        "case_id": case_id_for(args),
        "method": args.method,
        "block": args.block,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "hidden": args.hidden,
        "n_heads": args.heads,
        "head_dim": args.hidden // args.heads if args.heads else None,
        "intermediate": args.intermediate,
        "world_size": world,
        "dtype": args.dtype,
        "head_bucket": (args.head_bucket or None) if args.method == "tsp" and args.block == "attn" else None,
        "overlap": bool(args.overlap) if args.method == "tsp" else None,
        "attn_backend": args.attn_backend if args.block == "attn" else None,
        "seed": args.seed,
        "trial": args.trial_index,
        "warmup": args.warmup,
        "iters": args.iters,
        "launch_cmd": os.environ.get("TSPFOLD_LAUNCH_CMD", " ".join([sys.executable, *sys.argv])),
    }


def build_case(args, shape: BlockShape, rank: int, world: int, device: torch.device):
    """Create the module, its persistent shard, and the rank's input. Global tensors stay on CPU."""
    dtype = parse_dtype(args.dtype)
    if args.block == "attn":
        check_attn_backend(args.attn_backend)
        bucket = shape.check_head_bucket(args.head_bucket or None)
        packed = shard_attention(make_attention_weights(args.hidden, args.seed, dtype), rank, world, args.heads)
    else:
        packed = shard_mlp(make_mlp_weights(args.hidden, args.intermediate, args.seed, dtype), rank, world)
    if args.method == "tsp":  # generate only this rank's zigzag chunks of the global input
        sc = shape.chunk_len
        x = torch.cat(
            [make_input_range(args.batch, args.hidden, args.seed, c * sc, (c + 1) * sc, dtype)
             for c in zigzag_chunk_ids(rank, world)],
            dim=1,
        )  # fmt: skip
    else:
        x = make_input(args.batch, args.seq_len, args.hidden, args.seed, dtype)
    packed, x = packed.to(device), x.to(device).contiguous()
    if args.method == "tp":
        module = (
            TPAttention(packed, shape, backend=args.attn_backend) if args.block == "attn" else TPMLP(packed, shape)
        )
    elif args.block == "attn":
        module = TSPAttention(packed, shape, head_bucket=bucket, overlap=bool(args.overlap), backend=args.attn_backend)
    else:
        module = TSPMLP(packed, shape, overlap=bool(args.overlap))
    return module, x, packed


def timed_iterations(module, x: torch.Tensor, iters: int, device: torch.device) -> list[float]:
    times = []
    for _ in range(iters):
        dist.barrier()
        synchronize(device)
        t0 = time.perf_counter()
        y = module(x)
        synchronize(device)
        times.append(time.perf_counter() - t0)
        del y
    return times


def _stats_ms(values_s: list[float]) -> dict:
    ms = sorted(v * 1e3 for v in values_s)
    q = statistics.quantiles(ms, n=10, method="inclusive") if len(ms) > 1 else [ms[0]] * 9
    return {
        "median": statistics.median(ms),
        "mean": statistics.fmean(ms),
        "std": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "min": ms[0],
        "max": ms[-1],
        "p10": q[0],
        "p90": q[-1],
    }


def memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "device_used": total - free,  # whole device: CUDA context, NCCL buffers, other processes
        "device_total": total,
        "alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


def kernel_category(name: str) -> str:
    n = name.lower()
    if "nccl" in n or "gloo" in n or "c10d::" in n:
        return "comm"
    if any(s in n for s in ("flash", "fmha", "attention", "sdpa", "softmax")):
        return "attention"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "cublas", "gemv", "aten::mm", "aten::addmm", "aten::bmm", "matmul")):
        return "gemm"
    if any(s in n for s in ("copy", "cat", "elementwise", "silu", "mul", "fill", "memcpy", "memset", "reduce")):
        return "elementwise_copy"
    return "other"


def profile_breakdown(module, x: torch.Tensor, device: torch.device, n: int) -> dict:
    """Per-iteration self time by category (device kernels on CUDA, aten ops on CPU) + attention evidence."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(n):
            y = module(x)
            synchronize(device)
            del y
    cats: dict[str, float] = {}
    kernels, attn_ops = [], {}
    for evt in prof.key_averages():
        on_device = evt.device_type == DeviceType.CUDA
        if "scaled_dot_product" in evt.key or "attention" in evt.key.lower() or "flash" in evt.key.lower():
            if not on_device:
                attn_ops[evt.key] = evt.count / n
        if device.type == "cuda" and not on_device:
            continue
        us = (evt.self_device_time_total if on_device else evt.self_cpu_time_total) / n
        if us <= 0:
            continue
        cat = kernel_category(evt.key)
        cats[cat] = cats.get(cat, 0.0) + us
        kernels.append({"name": evt.key[:160], "category": cat, "us_per_iter": us, "calls_per_iter": evt.count / n})
    kernels.sort(key=lambda k: -k["us_per_iter"])
    return {
        "iters": n,
        "unit": "device kernel self time" if device.type == "cuda" else "cpu op self time",
        "category_us_per_iter": cats,
        "attention_kernels": sorted({k["name"] for k in kernels if k["category"] == "attention"}),
        "attention_ops": attn_ops,
        "top_kernels": kernels[:25],
    }


def output_checksum(module, x: torch.Tensor, method: str) -> float:
    """Sum of the full global output in fp64, accumulated in row chunks to avoid an fp64 copy.

    TSP ranks hold disjoint tokens (summed across ranks); TP ranks hold the same tokens (rank-local sum).
    """
    y = module(x)
    total = sum(chunk.double().sum().item() for chunk in y.reshape(-1, y.size(-1)).split(4096))
    s = torch.tensor([total], dtype=torch.float64, device=x.device)
    if method == "tsp":
        dist.all_reduce(s)
    del y
    return s.item()


def run(args, rank: int, world: int, device: torch.device) -> list[dict]:
    rec = base_record(args, world)
    dtype = parse_dtype(args.dtype)
    shape = BlockShape(args.batch, args.seq_len, args.hidden, args.heads, args.intermediate, world)
    rec["precision"] = set_fp32_strict(bool(args.fp32_strict))
    beta = torch.empty((), dtype=dtype).element_size()
    rec["params"] = {
        "block_total": shape.attn_params if args.block == "attn" else shape.mlp_params,
        "per_rank": (shape.attn_params if args.block == "attn" else shape.mlp_params) // world,
    }
    rec["theory"] = {
        "comm_bytes_per_rank": comm_bytes(args.method, args.block, shape, beta),
        "persistent_weight_bytes": persistent_bytes(args.block, shape, beta),
        "activation_io_bytes": activation_io_bytes(args.method, shape, beta),
    }

    module, x, packed = build_case(args, shape, rank, world, device)
    rec["weight_shard_bytes"] = packed.numel() * packed.element_size()
    rec["input_bytes"] = x.numel() * x.element_size()
    dist.barrier()
    fault = os.environ.get("TSPFOLD_INJECT_FAULT", "")  # test hook, e.g. "oom@rank1"
    for i in range(args.warmup):
        # Same pattern as a measured iteration: unsynchronized warmups would pipeline forwards and grow
        # the allocator cache (NCCL-stream buffers are not reusable until those streams finish).
        dist.barrier()
        synchronize(device)
        y = module(x)
        synchronize(device)
        del y
        if i == 0 and fault == f"oom@rank{rank}":
            raise torch.OutOfMemoryError("injected OOM (TSPFOLD_INJECT_FAULT)")

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = memory_snapshot(device)
    trials = [timed_iterations(module, x, args.iters, device) for _ in range(args.trials)]
    after = memory_snapshot(device)

    # ---- everything below is outside the timed and memory-measured region ----
    rec["output_checksum"] = output_checksum(module, x, args.method)
    per_rank = [None] * world
    dist.all_gather_object(per_rank, {"trials": trials, "before": before, "after": after})
    profile = profile_breakdown(module, x, device, args.profile) if args.profile else None
    profiles = [None] * world
    dist.all_gather_object(profiles, profile)
    if rank != 0:
        return []

    def mem_max(key: str, snap: str):
        vals = [r[snap].get(key) for r in per_rank]
        return {"max": max(vals), "per_rank": vals} if all(v is not None for v in vals) else None

    rec["memory"] = {
        "baseline_allocated": mem_max("allocated", "before"),
        "baseline_reserved": mem_max("reserved", "before"),
        "peak_allocated": mem_max("peak_allocated", "after"),
        "peak_reserved": mem_max("peak_reserved", "after"),
        "device_used_after": mem_max("device_used", "after"),
        "device_total": mem_max("device_total", "after"),
        "alloc_retries": mem_max("alloc_retries", "after"),
    }
    if profiles[0] is not None:
        rec["profile"] = {f"rank{r}": p for r, p in enumerate(profiles)}
    rec["env"] = collect_env(device)
    out = []
    for t in range(args.trials):
        iter_max = [max(r["trials"][t][i] for r in per_rank) for i in range(args.iters)]
        lat = _stats_ms(iter_max)
        out.append(
            {
                **rec,
                "status": "ok",
                "error": None,
                "inproc_trial": t,
                "latency_ms": lat,
                "latency_ms_iters": [v * 1e3 for v in iter_max],
                "rank_median_ms": [statistics.median(r["trials"][t]) * 1e3 for r in per_rank],
                "throughput_tok_s": args.batch * args.seq_len / (lat["median"] / 1e3),
            }
        )
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, torch.OutOfMemoryError):
        return "oom"
    if isinstance(exc, ConfigError):
        return "invalid_config"
    msg = str(exc)
    if "No available kernel" in msg or "not supported" in msg.lower():
        return "unsupported"
    return "error"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, world, device = init_from_env(args.device, args.pg_timeout)
    cid = case_id_for(args)
    status_dir = Path(args.status_dir) if args.status_dir else Path(args.out).parent / "status"
    status_path = status_dir / f"{cid}.rank{rank}.status.json"
    try:
        records = run(args, rank, world, device)
    except BaseException as exc:  # noqa: BLE001 - classify, persist, then exit without collectives
        status = classify_error(exc)
        info = {"case_id": cid, "rank": rank, "status": status, "error": f"{type(exc).__name__}: {exc}"[:4000]}
        if device.type == "cuda":
            try:
                info["memory"] = memory_snapshot(device)
            except Exception:  # noqa: BLE001
                pass
        _write_json(status_path, info)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(2 if status == "invalid_config" else 3)
    _write_json(status_path, {"case_id": cid, "rank": rank, "status": "ok"})
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        for r in records:
            print(
                f"{cid} trial={r['inproc_trial']} median={r['latency_ms']['median']:.3f} ms "
                f"thr={r['throughput_tok_s']:.4g} tok/s"
                + (f" peak_alloc={r['memory']['peak_allocated']['max'] / 2**20:.1f} MiB" if r["memory"]["peak_allocated"] else "")
            )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
