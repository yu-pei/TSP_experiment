"""Hardware gate (spec section 6 / task T02).

  NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m tspfold.hwcheck --out-dir results/env 2>&1 | tee results/env/hwcheck.log

Writes env_manifest.json, nvidia_smi_topo.txt, nvidia_smi_nvlink.txt and hw_gate.json. The gate passes
only if CUDA is used, the two visible GPUs are NVLink-connected in `nvidia-smi topo -m`, peer access
is available, and value-checked collective/P2P operations succeed. Bandwidth numbers are reported as
corroborating evidence; NCCL_DEBUG=INFO output in the log shows the transport NCCL selected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .dist_utils import all_gather_stacked, init_from_env, synchronize
from .env import collect_env, parse_topology, run_cmd, visible_to_physical
from .tsp import ring_exchange


def collective_sanity(device: torch.device) -> dict:
    rank, D = dist.get_rank(), dist.get_world_size()
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        x = torch.full((4096,), float(rank + 1), dtype=dtype, device=device)
        y = x.clone()
        dist.all_reduce(y)
        results[f"all_reduce_{name}"] = bool((y == D * (D + 1) / 2).all())
        ok = True
        for src in range(D):
            b = x.clone()
            dist.broadcast(b, src=src)
            ok &= bool((b == src + 1).all())
        results[f"broadcast_{name}"] = ok
        g = all_gather_stacked(x)
        results[f"all_gather_{name}"] = all(bool((g[r] == r + 1).all()) for r in range(D))
        recv = torch.empty_like(x)
        for req in ring_exchange(x, recv):
            req.wait()
        results[f"ring_p2p_{name}"] = bool((recv == (rank - 1) % D + 1).all())
    flags = torch.tensor([int(v) for v in results.values()], device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return {k: bool(v) for k, v in zip(results, flags.tolist())}


def _time(fn, device: torch.device, iters: int) -> float:
    for _ in range(3):
        fn()
    synchronize(device)
    times = []
    for _ in range(iters):
        dist.barrier()
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    t = torch.tensor([statistics.median(times)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def bandwidth(device: torch.device, sizes_mib: list[int], iters: int) -> list[dict]:
    """Bandwidth in GB/s (1e9 B/s). busbw follows nccl-tests conventions."""
    rank, D = dist.get_rank(), dist.get_world_size()

    def one_way(buf, recv):
        if rank == 0:
            dist.send(buf, dst=1)
        elif rank == 1:
            dist.recv(recv, src=0)

    def exchange(buf, recv):
        for req in ring_exchange(buf, recv):
            req.wait()

    def all_reduce(buf, recv):
        dist.all_reduce(buf)

    def all_gather(buf, recv):
        all_gather_stacked(buf)

    rows = []
    for mib in sizes_mib:
        buf = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device=device)
        recv = torch.empty_like(buf)
        nbytes = buf.numel() * buf.element_size()
        t_p2p, t_x, t_ar, t_ag = (
            _time(lambda op=op, b=buf, r=recv: op(b, r), device, iters)
            for op in (one_way, exchange, all_reduce, all_gather)
        )
        rows.append(
            {
                "size_mib": mib,
                "p2p_unidirectional_GBps": nbytes / t_p2p / 1e9,
                "ring_exchange_per_direction_GBps": nbytes / t_x / 1e9,
                "all_reduce_busbw_GBps": nbytes / t_ar / 1e9 * 2 * (D - 1) / D,
                "all_gather_busbw_GBps": D * nbytes / t_ag / 1e9 * (D - 1) / D,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/env")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sizes-mib", default="16,64,256,1024")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)
    rank, world, device = init_from_env(args.device)
    out = Path(args.out_dir)
    sanity = collective_sanity(device)
    bw = bandwidth(device, [int(s) for s in args.sizes_mib.split(",")], args.iters) if world == 2 else []
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        env = collect_env(device)
        (out / "env_manifest.json").write_text(json.dumps(env, indent=2))
        topo = run_cmd(["nvidia-smi", "topo", "-m"]) or ""
        (out / "nvidia_smi_topo.txt").write_text(topo or "nvidia-smi topo -m unavailable\n")
        (out / "nvidia_smi_nvlink.txt").write_text(run_cmd(["nvidia-smi", "nvlink", "--status"]) or "unavailable\n")
        gate: dict = {"world_size": world, "device_type": device.type, "collectives": sanity, "bandwidth": bw}
        reasons = []
        if device.type != "cuda":
            reasons.append("CUDA not used")
        if world != 2:
            reasons.append(f"world size {world} != 2")
        if device.type == "cuda":
            phys = visible_to_physical()
            gate["visible_to_physical"] = phys
            link = None
            if len(phys) >= 2 and None not in phys[:2]:
                link = parse_topology(topo).get((phys[0], phys[1]))
            gate["topology_link_gpu0_gpu1"] = link
            gate["nvlink_topology"] = bool(link and link.startswith("NV"))
            if not gate["nvlink_topology"]:
                reasons.append(f"visible GPUs 0/1 not NVLink-connected in topo (link={link!r})")
            gate["peer_access"] = torch.cuda.device_count() >= 2 and torch.cuda.can_device_access_peer(0, 1)
            if not gate["peer_access"]:
                reasons.append("no CUDA peer access between visible GPUs 0 and 1")
        if not all(sanity.values()):
            reasons.append("collective/P2P sanity failed")
        gate["gate_pass"] = not reasons
        gate["reasons"] = reasons
        (out / "hw_gate.json").write_text(json.dumps(gate, indent=2))
        print(json.dumps(gate, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
