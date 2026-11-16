"""Benchmark sweep driver: one fresh `torchrun` process group per (trial, block, B, S, method, variant).

  python -m tspfold.sweep --config configs/sweep_main.json [--results-dir results] [--dry-run]

Layout: <results>/<sweep name>/{raw/<case>.jsonl, logs/<case>.log, status/, commands.txt, sweep_config.json}
Each case is bounded by a wall-clock timeout; the whole launch process group is killed on expiry.
Outcomes are recorded as distinct statuses: ok, oom, unsupported, invalid_config, error, crash, timeout,
skipped (S at or beyond a size that already ran out of memory for the same method/block/batch/variant).
Re-running the same command resumes: cases with a terminal record are not launched again.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .bench import base_record, build_parser, case_id_for
from .launch import torchrun_command

TERMINAL = {"ok", "oom", "unsupported", "invalid_config", "skipped"}


def expand_cases(cfg: dict) -> list[dict]:
    """Ordered case list. Trials are the outer loop and TP/TSP order alternates per trial."""
    points = []
    for grid in cfg["grid"]:
        for B, S in itertools.product(grid["batch"], grid["seq_len"]):
            if (B, S) not in points:
                points.append((B, S))
    points.sort(key=lambda p: (p[0], p[1]))
    tsp_variants = cfg.get("tsp_variants", [{"overlap": 1, "head_bucket": 0}])
    cases = []
    for trial in range(cfg.get("trials", 1)):
        methods = cfg["methods"] if trial % 2 == 0 else list(reversed(cfg["methods"]))
        for block in cfg["blocks"]:
            for B, S in points:
                for method in methods:
                    variants = tsp_variants if method == "tsp" else [{"overlap": 1, "head_bucket": 0}]
                    for var in variants:
                        if block == "mlp" and var.get("head_bucket", 0):
                            continue  # head bucketing is attention-only
                        cases.append(
                            {
                                "method": method,
                                "block": block,
                                "batch": B,
                                "seq_len": S,
                                "hidden": cfg["hidden"],
                                "heads": cfg["heads"],
                                "intermediate": cfg["intermediate"],
                                "dtype": cfg.get("dtype", "bf16"),
                                "head_bucket": var.get("head_bucket", 0),
                                "overlap": var.get("overlap", 1),
                                "attn_backend": cfg.get("attn_backend", "sdpa"),
                                "warmup": cfg.get("warmup", 10),
                                "iters": cfg.get("iters", 30),
                                "seed": cfg.get("seed", 0),
                                "trial_index": trial,
                                "profile": cfg.get("profile_iters", 0) if trial in cfg.get("profile_trials", []) else 0,
                            }
                        )
    # De-duplicate (e.g. mlp cases expanded from attention-only variants).
    seen, unique = set(), []
    for c in cases:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def bench_args(case: dict, out: Path, status_dir: Path, device: str) -> list[str]:
    args = [
        "--method", case["method"], "--block", case["block"],
        "--batch", str(case["batch"]), "--seq-len", str(case["seq_len"]),
        "--hidden", str(case["hidden"]), "--heads", str(case["heads"]), "--intermediate", str(case["intermediate"]),
        "--dtype", case["dtype"], "--head-bucket", str(case["head_bucket"]), "--overlap", str(case["overlap"]),
        "--attn-backend", case["attn_backend"], "--warmup", str(case["warmup"]), "--iters", str(case["iters"]),
        "--seed", str(case["seed"]), "--trial-index", str(case["trial_index"]), "--profile", str(case["profile"]),
        "--device", device, "--out", str(out), "--status-dir", str(status_dir),
    ]  # fmt: skip
    return args


def oom_key(case: dict) -> tuple:
    return (case["method"], case["block"], case["batch"], case["head_bucket"], case["overlap"], case["dtype"])


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(statuses: list[dict], returncode: int | None, timed_out: bool) -> tuple[str, str | None]:
    if timed_out:
        return "timeout", "wall-clock timeout"
    for kind in ("oom", "invalid_config", "unsupported", "error"):
        hit = [s for s in statuses if s.get("status") == kind]
        if hit:
            return kind, hit[0].get("error")
    return "crash", f"launcher exit code {returncode} without a result record"


def run_case(case: dict, sweep_dir: Path, device: str, timeout_s: float, nproc: int) -> dict:
    parsed = build_parser().parse_args(bench_args(case, Path("x"), Path("x"), device))
    cid = case_id_for(parsed)
    raw = sweep_dir / "raw" / f"{cid}.jsonl"
    status_dir = sweep_dir / "status"
    log_path = sweep_dir / "logs" / f"{cid}.log"
    for p in (raw.parent, status_dir, log_path.parent):
        p.mkdir(parents=True, exist_ok=True)
    for old in status_dir.glob(f"{cid}.rank*.status.json"):
        old.unlink()
    raw.unlink(missing_ok=True)

    cmd = torchrun_command("tspfold.bench", bench_args(case, raw, status_dir, device), nproc=nproc)
    cmd_str = shlex.join(cmd)
    with open(sweep_dir / "commands.txt", "a") as f:
        f.write(cmd_str + "\n")
    env = {**os.environ, "TSPFOLD_LAUNCH_CMD": cmd_str}
    t0 = time.time()
    timed_out = False
    with open(log_path, "w") as log:
        log.write(f"$ {cmd_str}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    elapsed = time.time() - t0
    records = read_records(raw)
    if records and not timed_out and proc.returncode == 0:
        return {"case_id": cid, "status": "ok", "elapsed_s": elapsed, "records": records}

    statuses = [json.loads(p.read_text()) for p in sorted(status_dir.glob(f"{cid}.rank*.status.json"))]
    status, error = classify(statuses, proc.returncode, timed_out)
    log_tail = log_path.read_text().splitlines()[-40:]
    rec = {
        **base_record(parsed, nproc),
        "case_id": cid,
        "status": status,
        "error": error,
        "rank_status": statuses,
        "returncode": proc.returncode,
        "launch_cmd": cmd_str,
        "log_tail": log_tail,
        "elapsed_s": elapsed,
    }
    with open(raw, "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"case_id": cid, "status": status, "elapsed_s": elapsed, "records": [rec]}


def skipped_record(case: dict, sweep_dir: Path, device: str, nproc: int, bound: int) -> dict:
    parsed = build_parser().parse_args(bench_args(case, Path("x"), Path("x"), device))
    rec = {
        **base_record(parsed, nproc),
        "status": "skipped",
        "error": f"not launched: same method/block/batch/variant ran out of memory at seq_len={bound}",
    }
    raw = sweep_dir / "raw" / f"{rec['case_id']}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(rec) + "\n")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="print the launch commands without running them")
    ap.add_argument("--retry-failed", action="store_true", help="re-launch cases whose last status was error/crash/timeout")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    sweep_dir = Path(args.results_dir) / cfg["name"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_config.json").write_text(json.dumps(cfg, indent=2))
    cases = expand_cases(cfg)
    timeout_s = float(cfg.get("timeout_s", 1200))
    print(f"sweep {cfg['name']}: {len(cases)} cases -> {sweep_dir}", flush=True)

    oom_bound: dict[tuple, int] = {}
    summary = {}
    for i, case in enumerate(cases, 1):
        parsed = build_parser().parse_args(bench_args(case, Path("x"), Path("x"), args.device))
        cid = case_id_for(parsed)
        existing = read_records(sweep_dir / "raw" / f"{cid}.jsonl")
        last = existing[-1]["status"] if existing else None
        if args.dry_run:
            print(shlex.join(torchrun_command("tspfold.bench", bench_args(case, Path("<raw>"), Path("<status>"), args.device), nproc=args.nproc)))
            continue
        if last == "oom":
            oom_bound[oom_key(case)] = min(oom_bound.get(oom_key(case), case["seq_len"]), case["seq_len"])
        if last in TERMINAL or (last is not None and not args.retry_failed):
            summary[last] = summary.get(last, 0) + 1
            continue
        bound = oom_bound.get(oom_key(case))
        if bound is not None and case["seq_len"] >= bound:
            skipped_record(case, sweep_dir, args.device, args.nproc, bound)
            summary["skipped"] = summary.get("skipped", 0) + 1
            print(f"[{i}/{len(cases)}] skipped {cid} (oom at S={bound})", flush=True)
            continue
        result = run_case(case, sweep_dir, args.device, timeout_s, args.nproc)
        status = result["status"]
        summary[status] = summary.get(status, 0) + 1
        if status == "oom":
            oom_bound[oom_key(case)] = min(oom_bound.get(oom_key(case), case["seq_len"]), case["seq_len"])
        msg = ""
        if status == "ok":
            r = result["records"][0]
            msg = f"median={r['latency_ms']['median']:.3f} ms"
        print(f"[{i}/{len(cases)}] {status} {cid} {msg} ({result['elapsed_s']:.0f}s)", flush=True)
    print(f"done: {summary}", flush=True)
    return 0 if not set(summary) - TERMINAL else 1


if __name__ == "__main__":
    sys.exit(main())
