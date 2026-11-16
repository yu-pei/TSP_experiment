# GPU runbook (tasks T02, T08, T10-T14)

Everything below runs from the repository root on the two-GPU host. Keep the git working tree clean
during measurements so every record's `env.git.rev` identifies the exact source (`git.dirty` must be
`false`).

## 0. Environment

```bash
uv sync --extra analysis
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
nvidia-smi                      # GPUs must be idle: no other processes (memory and timing isolation)
nvidia-smi topo -m              # find an NVLink-connected pair (entry NV# between them)
export CUDA_VISIBLE_DEVICES=0,1 # set to that pair; keep it for every step below
make test                       # CPU suite must pass on this host too
```

If `torch.cuda.is_available()` is false or the driver is too old for cu126, fix the environment first
(see README "Setup"); record any index change.

## 1. Hardware gate (T02)

```bash
make hw-check
cat results/env/hw_gate.json
grep -E "via (P2P|NVLS|SHM|NET)" results/env/hwcheck.log | head
```

Proceed only if `gate_pass` is `true`: the visible pair is `NV#` in `nvidia-smi topo -m`, peer access
is available, and all value-checked collectives pass. The log shows the NCCL transport per channel; a
`P2P` transport with NVLink topology plus NVLink-class bandwidth in `hw_gate.json` corroborates the
gate. A PCIe-only pair does **not** satisfy the hardware requirement; do not label such runs as such.

## 2. Correctness on GPUs (T08)

```bash
make correctness-gpu
```

Acceptance: every record in `results/correctness/gpu_d2.jsonl` and `smoke_d1.jsonl` has
`passed: true` (thresholds: FP32 atol 1e-5 / rtol 1e-4 with TF32 off; BF16 atol 2e-2 / rtol 2e-2).
Each record also reports `max_abs_err`, `l2_rel_err`, `worst_tol_ratio`, repeated-call consistency,
TP cross-rank equality, and future-token perturbation checks. Compare each method's error with the
`unsharded` record of the same case (same kernels and dtype on one GPU) to separate precision effects
from distribution effects. If anything fails, investigate and document before benchmarking; changing
a threshold requires a written justification in the report.

## 3. Sweeps (T10, T11)

```bash
make gpu-smoke        # ~2 min, writes to scratch/: every method/block/variant must report ok
make sweep-main       # 96 launches (3 trials x 2 blocks x 8 (B,S) points x 2 methods)
make sweep-ablation   # 36 launches, TSP only: overlap on/off, head buckets 4 and 1
make sweep-long       # 20 launches, B=1 up to S=1,048,576; OOMs recorded, larger S skipped
```

- Re-running a target resumes; add `--retry-failed` (edit the target or run `python -m tspfold.sweep
  ... --retry-failed`) to relaunch `error|crash|timeout` cases after a fix.
- Per-case logs: `results/<sweep>/logs/<case>.log`; commands: `results/<sweep>/commands.txt`.
- Check `alloc_retries` in records; non-zero values mean allocator pressure affected timing.
- Overlap/memory review (T10): compare the ablation's `ov0` vs `ov1` latencies and the profiler's
  `category_us_per_iter` (comm vs gemm vs attention); confirm TSP peak memory includes at most one
  temporary weight shard and the bucketed K/V buffers expected from `docs/DESIGN.md`.

## 4. Analysis (T12)

```bash
make analyze
```

Outputs: `results/analysis/summary.md`, `summary.csv`, `matched.csv` (speedup = TP/TSP latency,
memory reduction = 1 - TSP/TP peak allocated, theoretical comm ratio), `outcomes.csv`, and figures
`<sweep>_seq_sweep.{png,pdf}`, `<sweep>_batch_sweep.{png,pdf}`. Check that `checksum_rel_diff` in
`matched.csv` is small (TP and TSP computed the same operator at benchmark scale) and that
`attention_kernels` names the backend each method actually used.

## 5. Report and packaging (T13, T14)

- Fill the measured sections of `report/report.tex` from `results/analysis` only; `make report`
  enforces the two-page limit (needs `latexmk` and `pdfinfo`).
- Commit `results/` (raw records, logs, env, correctness, analysis), the report PDF, and exported AI
  logs (`ai_logs/`), after reviewing logs for credentials or private data.
