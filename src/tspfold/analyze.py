"""Aggregate raw benchmark records into tables and figures (task T12).

  python -m tspfold.analyze --results-dir results --out-dir results/analysis

Aggregation (identical for every method): per independent trial the latency is the median over its
iterations of the per-iteration MAX across ranks; a configuration's latency is the median of its trial
medians, with min/max over trials as variability. Throughput = B*S / latency (never multiplied by D).
Matched ratios use the same sweep, block, B, S, model shape, dtype and attention backend:
  speedup = TP latency / TSP latency;  memory reduction = 1 - TSP peak allocated / TP peak allocated.
Theoretical communication volumes are reported in separate, labelled columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

KEY = ("sweep", "block", "method", "batch", "seq_len", "hidden", "n_heads", "intermediate", "dtype",
       "head_bucket", "overlap", "attn_backend", "world_size")  # fmt: skip
MATCH = ("sweep", "block", "batch", "seq_len", "hidden", "n_heads", "intermediate", "dtype", "attn_backend", "world_size")
COLORS = {"tp": "#1f77b4", "tsp": "#ff7f0e"}
MiB = 2**20


def load_records(results_dir: Path) -> list[dict]:
    recs = []
    for path in sorted(results_dir.glob("*/raw/*.jsonl")):
        sweep = path.parent.parent.name
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["sweep"] = sweep
                recs.append(rec)
    return recs


def _sort_key(key: tuple) -> tuple:
    return tuple((0, x, "") if isinstance(x, (int, float)) else (1, 0, str(x)) for x in key)


def _mem(rec: dict, key: str) -> float | None:
    m = (rec.get("memory") or {}).get(key)
    return m["max"] if m else None


def aggregate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in KEY)].append(r)
    rows, outcomes = [], []
    for key, recs in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        ok = [r for r in recs if r["status"] == "ok"]
        bad = [r for r in recs if r["status"] != "ok"]
        base = dict(zip(KEY, key))
        for r in bad:
            outcomes.append({**base, "trial": r.get("trial"), "status": r["status"], "error": (r.get("error") or "")[:200]})
        if not ok:
            continue
        trial_medians = [r["latency_ms"]["median"] for r in ok]
        lat = statistics.median(trial_medians)
        r0 = ok[0]
        prof = (r0.get("profile") or {}).get("rank0")
        rows.append(
            {
                **base,
                "n_trials": len(ok),
                "latency_ms": lat,
                "latency_ms_min_trial": min(trial_medians),
                "latency_ms_max_trial": max(trial_medians),
                "trial_cv": statistics.pstdev(trial_medians) / statistics.fmean(trial_medians) if len(ok) > 1 else 0.0,
                "throughput_tok_s": r0["batch"] * r0["seq_len"] / (lat / 1e3),
                "peak_alloc_MiB": _ratio_or_none(max(filter(None, (_mem(r, "peak_allocated") for r in ok)), default=None), MiB),
                "peak_reserved_MiB": _ratio_or_none(max(filter(None, (_mem(r, "peak_reserved") for r in ok)), default=None), MiB),
                "baseline_alloc_MiB": _ratio_or_none(_mem(r0, "baseline_allocated"), MiB),
                "device_used_MiB": _ratio_or_none(_mem(r0, "device_used_after"), MiB),
                "alloc_retries": max((_mem(r, "alloc_retries") or 0) for r in ok),
                "params_block": (r0.get("params") or {}).get("block_total"),
                "theory_comm_MiB_per_rank": _ratio_or_none(r0["theory"]["comm_bytes_per_rank"]["total"], MiB),
                "output_checksum": r0.get("output_checksum"),
                "attention_kernels": ";".join((prof or {}).get("attention_kernels", [])),
                "profile_us": json.dumps((prof or {}).get("category_us_per_iter", {})),
                "gpu": ((r0.get("env") or {}).get("visible_gpus") or [{}])[0].get("name"),
            }
        )
    return rows, outcomes


def _ratio_or_none(v, d):
    return None if v is None else v / d


def is_default_tsp(row: dict) -> bool:
    return row["method"] == "tsp" and row["overlap"] in (True, None) and row["head_bucket"] in (None,)


def matched(rows: list[dict]) -> list[dict]:
    tp = {tuple(r[k] for k in MATCH): r for r in rows if r["method"] == "tp"}
    out = []
    for r in rows:
        if not is_default_tsp(r):
            continue
        t = tp.get(tuple(r[k] for k in MATCH))
        if t is None:
            continue
        out.append(
            {
                **{k: r[k] for k in MATCH},
                "tp_latency_ms": t["latency_ms"],
                "tsp_latency_ms": r["latency_ms"],
                "speedup_tp_over_tsp": t["latency_ms"] / r["latency_ms"],
                "tp_peak_alloc_MiB": t["peak_alloc_MiB"],
                "tsp_peak_alloc_MiB": r["peak_alloc_MiB"],
                "memory_reduction": (1 - r["peak_alloc_MiB"] / t["peak_alloc_MiB"]) if t["peak_alloc_MiB"] else None,
                "theory_comm_ratio_tsp_over_tp": r["theory_comm_MiB_per_rank"] / t["theory_comm_MiB_per_rank"],
                "checksum_rel_diff": abs(r["output_checksum"] - t["output_checksum"]) / max(abs(t["output_checksum"]), 1e-12)
                if r["output_checksum"] is not None and t["output_checksum"] is not None
                else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def plot_all(rows: list[dict], pairs: list[dict], out_dir: Path, sweep: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    srows = [r for r in rows if r["sweep"] == sweep and (r["method"] == "tp" or is_default_tsp(r))]
    spairs = [p for p in pairs if p["sweep"] == sweep]
    if not srows:
        return []
    blocks = [b for b in ("attn", "mlp") if any(r["block"] == b for r in srows)]
    fig, axes = plt.subplots(3, len(blocks), figsize=(3.6 * len(blocks), 7.8), squeeze=False)
    for j, block in enumerate(blocks):
        for method in ("tp", "tsp"):
            pts = sorted((r for r in srows if r["block"] == block and r["method"] == method and r["batch"] == 1),
                         key=lambda r: r["seq_len"])  # fmt: skip
            if not pts:
                continue
            S = [r["seq_len"] for r in pts]
            lat = [r["latency_ms"] for r in pts]
            err = [[r["latency_ms"] - r["latency_ms_min_trial"] for r in pts], [r["latency_ms_max_trial"] - r["latency_ms"] for r in pts]]
            axes[0][j].errorbar(S, lat, yerr=err, marker="o", ms=3, capsize=2, color=COLORS[method], label=method.upper())
            mem = [r["peak_alloc_MiB"] for r in pts]
            if all(m is not None for m in mem):
                axes[1][j].plot(S, [m / 1024 for m in mem], marker="o", ms=3, color=COLORS[method], label=f"{method.upper()} peak")
                axes[1][j].plot(S, [r["baseline_alloc_MiB"] / 1024 for r in pts], ls="--", color=COLORS[method], alpha=0.6,
                                label=f"{method.upper()} pre-op")  # fmt: skip
        pp = sorted((p for p in spairs if p["block"] == block and p["batch"] == 1), key=lambda p: p["seq_len"])
        if pp:
            S = [p["seq_len"] for p in pp]
            axes[2][j].plot(S, [p["speedup_tp_over_tsp"] for p in pp], marker="o", ms=3, color="k", label="speedup TP/TSP")
            if all(p["memory_reduction"] is not None for p in pp):
                axes[2][j].plot(S, [1 + p["memory_reduction"] for p in pp], marker="s", ms=3, color="g",
                                label="1 + mem. reduction")  # fmt: skip
            axes[2][j].axhline(1.0, color="gray", lw=0.8)
        axes[0][j].set_title(f"{'Attention' if block == 'attn' else 'Gated MLP'} (B=1)")
        if not axes[1][j].lines:
            axes[1][j].text(0.5, 0.5, "allocator stats unavailable\n(non-CUDA run)", ha="center", va="center",
                            transform=axes[1][j].transAxes, fontsize=8)  # fmt: skip
        for i, ylabel in enumerate(["latency [ms]", "peak allocated [GiB]", "ratio"]):
            ax = axes[i][j]
            ax.set_xscale("log", base=2)
            if i == 0:
                ax.set_yscale("log")
            ax.set_xlabel("sequence length S")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, which="both")
            if ax.lines:
                ax.legend(fontsize=7)
    fig.suptitle(f"{sweep}: measured, 2 ranks", fontsize=9)
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{sweep}_seq_sweep.{ext}"
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)

    # Batch sweep at the most common fixed S with B > 1.
    multi = [r for r in srows if r["batch"] > 1]
    if multi:
        s_fixed = statistics.mode(r["seq_len"] for r in multi)
        fig, axes = plt.subplots(1, len(blocks), figsize=(3.6 * len(blocks), 2.8), squeeze=False)
        for j, block in enumerate(blocks):
            for method in ("tp", "tsp"):
                pts = sorted((r for r in srows if r["block"] == block and r["method"] == method and r["seq_len"] == s_fixed),
                             key=lambda r: r["batch"])  # fmt: skip
                if pts:
                    axes[0][j].plot([r["batch"] for r in pts], [r["throughput_tok_s"] for r in pts], marker="o",
                                    color=COLORS[method], label=method.upper())  # fmt: skip
            axes[0][j].set_title(f"{block} throughput, S={s_fixed}")
            axes[0][j].set_xlabel("batch B")
            axes[0][j].set_ylabel("tokens / s (global)")
            axes[0][j].grid(alpha=0.3)
            axes[0][j].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = out_dir / f"{sweep}_batch_sweep.{ext}"
            fig.savefig(p, dpi=200)
            paths.append(p)
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    records = load_records(Path(args.results_dir))
    rows, outcomes = aggregate(records)
    pairs = matched(rows)
    write_csv(out / "summary.csv", rows)
    write_csv(out / "matched.csv", pairs)
    write_csv(out / "outcomes.csv", outcomes)
    md = ["# Benchmark summary (generated by tspfold.analyze)", "", f"records: {len(records)}", ""]
    md += ["## Matched TP vs TSP", "", markdown_table(pairs, ["sweep", "block", "batch", "seq_len", "tp_latency_ms",
           "tsp_latency_ms", "speedup_tp_over_tsp", "tp_peak_alloc_MiB", "tsp_peak_alloc_MiB", "memory_reduction",
           "theory_comm_ratio_tsp_over_tp"]), ""]  # fmt: skip
    md += ["## All configurations", "", markdown_table(rows, ["sweep", "block", "method", "batch", "seq_len", "head_bucket",
           "overlap", "n_trials", "latency_ms", "trial_cv", "throughput_tok_s", "peak_alloc_MiB", "attention_kernels"]), ""]  # fmt: skip
    md += ["## Non-ok outcomes", "", markdown_table(outcomes, ["sweep", "block", "method", "batch", "seq_len", "trial", "status", "error"])]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    figs = []
    if not args.no_plots:
        for sweep in sorted({r["sweep"] for r in rows}):
            figs += plot_all(rows, pairs, out, sweep)
    print(f"{len(records)} records -> {len(rows)} configurations, {len(pairs)} matched pairs, {len(outcomes)} non-ok outcomes")
    for p in figs:
        print(f"figure: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
