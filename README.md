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
| T05-T07 TP baseline, TSP ring MLP, TSP attention | implemented; pass CPU/gloo and GPU/NCCL correctness |
| T08 correctness suite | **120 regression tests; 289 standard, 133 native-cuDNN, and 97 bucket GPU correctness records pass** |
| T09 benchmark runner, sweep driver, failure isolation | validated on GPU; injected single-rank OOM isolation tested on CPU |
| T02 hardware gate | **passed: two GB200 GPUs, NV18, peer access and value-checked collectives** |
| T10-T12 overlap/memory review, sweeps, analysis | **132 baseline + 54 backend + 63 bucket/control launches**, tables/plots saved; memory-limit sweep not run |
| T13-T14 report and packaging | [Markdown report](report/report.md) complete and print-checked at two pages; current AI transcript exported; PDF and earlier transcripts remain outstanding |

Original `sdpa` baseline measured on 2026-11-16 UTC with the NVIDIA container runtime documented in
[results/README.md](results/README.md). At B=1/S=32768 and the default BF16 model shape, TSP MLP
achieved **1.065x TP throughput with 40.8% less peak allocated memory**; default TSP attention was
**4.08x slower with 14.9% more peak allocated memory**. These are forward-block implementation
measurements, not training or decoding claims. Raw records, variability, backend details, and plots
are under `results/`.

The native-cuDNN backend correction reduces 32K-token TSP attention from **8.412 to 3.019 ms
(2.79x faster)** in fresh same-worker controls; TP with the same backend takes 2.142 ms. Peak
allocation is unchanged. Select `--attn-backend cudnn_frontend`; see
[the backend diagnosis and measurements](results/attention_dispatch/README.md).

With native cuDNN **and head bucketing together**, B=1/S=32768 attention saves **14.9% memory at
1.65x TP latency** (bucket 4), or **37.3% at 4.76x TP latency** (bucket 1). Same-run TSP MLP saves
40.8% memory with 1.041x TP throughput. See [the combined bucket/control experiment](results/native_bucketing/README.md).

## Layout

```
src/tspfold/
  config.py       shape contracts and validation (B, S, h, H, I, D)
  layout.py       zigzag shard/unshard/reorder, causal key bounds
  weights.py      deterministic global weights/inputs, packed head/channel shards
  ops.py          bottom-right causal attention dispatch, one SwiGLU shard
  cudnn_attention.py  optional native cuDNN bottom-right attention, bounded metadata-only plan cache
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
report/           report.md (measured report, <=2 printed pages); report.tex (legacy skeleton)
results/          env/, correctness/, <sweep>/{raw,logs}/, analysis/  (produced on the GPU host)
ai_logs/          exported AI chat logs (see ai_logs/README.md)
```

## Setup

Default locked environment: Python 3.12, `torch==2.14.1`, locked in `uv.lock`
(Linux x86_64 resolves `torch 2.14.1+cu126` from the PyTorch index; macOS arm64 uses the PyPI CPU build).

```bash
uv sync --extra analysis          # creates .venv from uv.lock
uv run pytest -q                  # CPU unit + gloo distributed tests (~15 s)
```

For the optional native cuDNN backend on supported Linux CUDA hosts:

```bash
uv sync --extra analysis --extra cudnn
```

`--attn-backend cudnn_frontend` uses cuDNN's actual bottom-right causal mask for both square and
rectangular attention, without allocating a dense mask or padding queries. It is forward-only and
requires CUDA FP16/BF16 inputs. The tested container already includes `nvidia-cudnn-frontend==1.23.0`;
do not reinstall its shared GPU packages. The existing `sdpa` default and PyTorch-only `cudnn`
restriction remain unchanged, so older measurements remain reproducible.

- The cu126 build supports Volta through Hopper with drivers >= 525. For Blackwell GPUs (or to use
  CUDA 13 with drivers >= 580), change the index URL in `pyproject.toml` to `.../whl/cu130` and run
  `uv lock` before `uv sync`; record the change.
- For ARM64/GB200 NVIDIA containers, preserve the preinstalled GPU stack rather than applying the
  x86_64 lock. Use an isolated system-site-packages venv and `PYTHONPATH=src` as described in
  [GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md#nvidia-containers-and-yolo-access). The measured container
  environment is recorded separately from `uv.lock`.
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
make sweep-attention-cudnn RESULTS=results/attention_dispatch  # optional native cuDNN comparison
make sweep-cudnn-bucketing RESULTS=results/native_bucketing    # buckets 8/4/2/1 plus MLP controls
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
