# TSP analysis: tensor+sequence parallel attention and gated MLP vs. 2-way TP

Forward-pass implementation and evaluation of the TSP schedules (Algorithms 1 and 2) from
*Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference*
(Shyam, Golubeva, Anthony; arXiv:2604.26294, **v1**), compared against conventional two-way tensor
parallelism on two NVLink-connected GPUs. The full project specification is
[TSP_PROJECT_SPEC.md](TSP_PROJECT_SPEC.md); design decisions and the algorithm-to-code mapping are in
[docs/DESIGN.md](docs/DESIGN.md).

## Status

| Spec task | State |
| --- | --- |
| T01 contracts and paper mapping | done: `config.py`, [docs/DESIGN.md](docs/DESIGN.md) |
| T03/T04 references, zigzag and weight helpers | done, unit-tested |
| T05-T07 TP baseline, TSP ring MLP, TSP attention | implemented; pass the CPU/gloo suite at D=1 and D=2 |
| T08 correctness suite | implemented; **CPU/gloo FP32+BF16 passing; GPU/NCCL run pending** |
| T09 benchmark runner, sweep driver, failure isolation | implemented; validated on CPU/gloo (timings meaningless) |
| T02 hardware gate | script ready (`make hw-check`); **needs the GPU host** |
| T10-T13 overlap/memory review, sweeps, analysis, report | analysis/plot code and report skeleton ready; **need GPU measurements** |

No performance numbers exist yet. Nothing in this repository should be read as a measured result
until `results/` contains GPU records produced by the commands below.

## Layout

```
src/tspfold/
  config.py       shape contracts and validation (B, S, h, H, I, D)
  layout.py       zigzag shard/unshard/reorder, causal key bounds
  weights.py      deterministic global weights/inputs, packed head/channel shards
  ops.py          bottom-right causal attention dispatch, one SwiGLU shard
  reference.py    unsharded float64 oracle
  tp.py           TP baseline (attention, MLP)
  tsp.py          TSP attention (Algorithm 1), TSP ring MLP (Algorithm 2)
  correctness.py  distributed correctness suite (pytest/gloo and torchrun/NCCL)
  bench.py        one benchmark case in a fresh process group
  sweep.py        sweep driver: per-case launch, timeout, outcome classification, resume
  hwcheck.py      hardware gate: topology/NVLink, peer access, collective sanity, bandwidth
  analyze.py      aggregation, matched ratios, figures
  theory.py       theoretical communication volumes (kept separate from measurements)
configs/          sweep definitions (main, long, ablation, cpu_smoke)
tests/            unit + distributed tests (CPU)
docs/             DESIGN.md, GPU_RUNBOOK.md, AI_USE.md
report/           report.tex (two-page report skeleton)
results/          env/, correctness/, <sweep>/{raw,logs}/, analysis/  (produced on the GPU host)
ai_logs/          exported AI chat logs (see ai_logs/README.md)
```

## Setup

Pinned environment: Python 3.12, `torch==2.14.1`, locked in `uv.lock`
(Linux x86_64 resolves `torch 2.14.1+cu126` from the PyTorch index; macOS arm64 uses the PyPI CPU build).

```bash
uv sync --extra analysis          # creates .venv from uv.lock
uv run pytest -q                  # CPU unit + gloo distributed tests (~15 s)
```

- The cu126 build supports Volta through Hopper with drivers >= 525. For Blackwell GPUs (or to use
  CUDA 13 with drivers >= 580), change the index URL in `pyproject.toml` to `.../whl/cu130` and run
  `uv lock` before `uv sync`; record the change.
- On Apple-silicon Macs with an x86_64 `uv`, create the venv with an arm64 interpreter first:
  `uv venv --python cpython-3.12-macos-aarch64-none && uv sync --extra analysis`.
- `torchrun --standalone` can hang on macOS (IPv6 reverse lookup); all commands here use static
  rendezvous on 127.0.0.1 instead (`tspfold.launch.torchrun_command`, Makefile `TORCHRUN`).

## Running

See [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md) for the full GPU procedure and acceptance checks.

```bash
make test             # local/CPU correctness
make hw-check         # T02 gate -> results/env/hw_gate.json
make correctness-gpu  # T08 on GPUs -> results/correctness/*.jsonl
make sweep-main       # T11 matched sweep -> results/main_bf16/raw/*.jsonl
make sweep-ablation   # overlap on/off, head buckets
make sweep-long       # extend S toward the memory limit
make analyze          # T12 -> results/analysis/{summary.md,matched.csv,*.png,*.pdf}
make report           # T13 -> report/report.pdf (page count checked)
```

Single benchmark case (what the sweep launches):

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29600 \
  -m tspfold.bench --method tsp --block attn --batch 1 --seq-len 8192 --out /tmp/case.jsonl
```

## Result records

Each benchmark record (`schema: tspfold.bench/1`) holds method, block, B/S/h/H/d/I, D, dtype,
head bucket, overlap mode, attention backend, seed, trial, per-iteration latencies (max over ranks),
latency statistics, global throughput `B*S/latency`, memory (pre-operation allocated baseline, peak
allocated/reserved and whole-device usage; max over ranks and per rank), theoretical communication
bytes, parameter counts, output checksum, optional profiler breakdown including the attention kernels
that ran, environment (GPU, driver, CUDA, NCCL, cuDNN, torch, git revision), the exact launch command,
and `status`/`error`. Failed launches produce records with `status` in
`oom | unsupported | invalid_config | error | crash | timeout | skipped`.

## AI use

This repository was produced with substantial AI assistance; see [docs/AI_USE.md](docs/AI_USE.md).
