# GPU runbook (tasks T02, T08, T10-T14)

Everything below runs from the repository root on the two-GPU host. Prefer a clean git working tree
so every record's `env.git.rev` identifies the exact source. If validating uncommitted fixes, preserve
the patch and source hashes below; `git.dirty: true` must be disclosed, not hidden or identified by
the base revision alone.

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

### NVIDIA containers and YOLO access

The GB200 validation used an **aarch64** NVIDIA container with Python 3.12.3, preinstalled PyTorch
`2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2, and NCCL 2.30.4. This is an explicitly recorded alternative
to the default x86_64/macOS lock, not a `uv.lock` reproduction. Do not run `uv sync` there or replace
the shared container's PyTorch installation.

From `~/repos/yolo`, wrap compound commands in an explicit remote shell:

```bash
uv run yolo ssh JOB_NAME -c "bash -lc 'hostname; uname -m; nvidia-smi'"
```

Without `bash -lc` and its quoting, shell separators in `-c` can execute later commands **locally**.
The non-interactive `Unable to use a TTY` warning is harmless when the remote command succeeds.
Use a separate checkout on the node; do not overwrite an existing working tree.

In that checkout, reuse the container's packages through an isolated venv:

```bash
uv venv --python /usr/bin/python --system-site-packages .venv
uv pip install --python .venv/bin/python pytest==9.2.0  # only if pytest is missing
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
make test
```

Use `.venv/bin/python` / `make`, not `uv run`, for subsequent commands in this container checkout.
NumPy and Matplotlib were already installed; exact distribution versions are in
`results/env/runtime_packages.json`. Only the isolated venv received additional packages.
`CUDA_VISIBLE_DEVICES=0,1` selects two GPUs even when the node exposes four. Recheck that the pair is
idle before measurements; leave the other GPUs and shared runtime unchanged.

For uncommitted fixes, save provenance **before** the sweeps and do not change runtime source during
measurement:

```bash
mkdir -p results/env
git diff --binary -- src tests Makefile pyproject.toml uv.lock configs > results/env/source.patch
sha256sum src/tspfold/*.py > results/env/source_sha256.txt
```

Reproduce the recorded base revision plus `source.patch`, verify the hashes, and use the recorded
container/package versions. Generated result files and later documentation edits are not runtime
source changes.

### Native cuDNN lower-right attention

The optional `cudnn_frontend` backend avoids the tested PyTorch 2.12 `CausalBias` path, which selects
FlashAttention for non-square lower-right attention even when cuDNN is available. The native backend
uses cuDNN's bottom-right diagonal directly and preserves strided/bucketed tensor layouts.
It supports forward-only CUDA FP16/BF16; unsupported inputs and missing dependencies fail explicitly.
Do not substitute `is_causal=True` on an ordinary non-square SDPA call: that changes the alignment.

On a standard locked Linux setup, enable it with `uv sync --extra analysis --extra cudnn`. On the
NVIDIA container, check the existing package rather than syncing/replacing the runtime:

```bash
.venv/bin/python -c "import cudnn; print(cudnn.__version__)"  # validated frontend: 1.23.0
.venv/bin/python -m pytest tests/test_ops.py -m gpu -q
.venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29600 -m tspfold.correctness \
  --suite gpu --block attn --dtype bf16 --attn-backend cudnn_frontend \
  --out results/attention_dispatch/correctness/cudnn_frontend_d2.jsonl
make sweep-attention-cudnn RESULTS=results/attention_dispatch
make analyze RESULTS=results/attention_dispatch
```

The correctness override applies to the suite's default attention cases; separately forced
Flash/efficient cases are omitted rather than duplicated. Normal correctness commands are unchanged.
Use a free rendezvous port if 29600 is occupied. Baseline `sdpa` and forced `flash` controls should
use the same shapes/trials and the same physical worker as the native-cuDNN run. Do not mix results
from before and after a worker replacement when attributing a speedup to the backend.

The original YOLO worker was replaced during the backend investigation. Historical source/results
were restored at `/tmp/tspfold-c92f5b03/repo`; new backend validation uses the separate checkout
`/tmp/tspfold-c92f5b03/kernel-dispatch`. Both use a private venv at `/tmp/tspfold-c92f5b03/venv`.
These node-local paths are ephemeral; keep completed artifacts locally and record a new hardware gate
after a replacement.

For the combined native-cuDNN memory/latency experiment, use
`make sweep-cudnn-bucketing RESULTS=results/native_bucketing`. This launches attention with all eight
local heads or buckets of 4, 2, and 1, plus same-worker MLP controls at the same three (B,S) points.
There are 63 fresh launches (45 attention, 18 MLP), with three independent trials per configuration.
`head_bucket` affects only attention, not MLP. Preserve the existing backend-comparison output root.

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
The command now exits nonzero when the gate fails, including when collectives succeed but NVLink
cannot be verified. GPU identity matching retains PCI domains (required on this GB200 node), and
topology parsing accepts nvidia-smi's ANSI-underlined headers.

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
- The profiler excludes user annotations, which otherwise duplicate NCCL kernel time, and recognizes
  the GB200 `nvjet` GEMM kernels. Its category times still overlap and are not additive wall time.
- `sdpa` is the same selection policy, not necessarily the same kernel: inspect both ranks'
  `profile.attention_ops` / `attention_kernels`. The measured container uses cuDNN for square attention
  and PyTorch FlashAttention for lower-right non-square attention.

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

- The current measured report is `report/report.md`, print-checked at two pages on A4/US Letter with
  11 pt text and 0.65-inch margins. It includes the user's role in directing the AI-assisted work.
  The legacy LaTeX source below is still a skeleton; `make report` does not render the Markdown file.
- Fill the measured sections of `report/report.tex` from `results/analysis` only; `make report`
  enforces the two-page limit (needs `latexmk` and `pdfinfo`).
- Commit `results/` (raw records, logs, env, correctness, analysis), the report PDF, and exported AI
  logs (`ai_logs/`), after reviewing logs for credentials or private data.
