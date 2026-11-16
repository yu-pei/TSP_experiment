# Results

Produced on the GPU host by the Makefile targets; commit everything here except `*/status/`.

**Publication privacy:** internal GPU hostnames and private network IPs are explicitly redacted in
the three NCCL hardware-check logs. Measurement values, GPU indices/PCI identities, source hashes,
and launch parameters are unchanged.

The historical `sdpa` main/ablation run is documented below. The later
[attention-dispatch investigation](attention_dispatch/README.md) fixes the dominant kernel-selection
issue and contains fresh same-worker SDPA/Flash/native-cuDNN comparisons: at B=1/S=32768, native cuDNN
reduces TSP latency from 8.412 to 3.019 ms without changing peak allocation. Each dataset has its own
source patch and hashes; the historical hashes identify that snapshot, not the current working tree.

The [combined native-cuDNN bucketing experiment](native_bucketing/README.md) confirms the memory
benefits with the efficient backend in the same configuration. At B=1/S=32768, bucket 4 saves 14.9%
versus TP with 1.65x latency; bucket 1 saves 37.3% with 4.76x latency. Same-run MLP saves 40.8% with
1.041x TP throughput. Thus both blocks save memory, but attention still trades latency for memory.

| Path | Producer | Content |
| --- | --- | --- |
| `env/` | `make hw-check` | `env_manifest.json`, `nvidia_smi_topo.txt`, `nvidia_smi_nvlink.txt`, `hw_gate.json`, `hwcheck.log` (NCCL_DEBUG=INFO) |
| `correctness/` | `make correctness-gpu` | `gpu_d2.jsonl`, `smoke_d1.jsonl` (+ logs): per case and method error metrics and checks |
| `<sweep>/raw/` | `make sweep-*` | one JSONL per launched case (`schema: tspfold.bench/1`), including failure outcomes |
| `<sweep>/logs/` | `make sweep-*` | full stdout/stderr per case; `<sweep>/commands.txt` lists exact launch commands |
| `analysis/` | `make analyze` | `summary.md/csv`, `matched.csv`, `outcomes.csv`, figures |

CPU/gloo runs (for example `configs/sweep_cpu_smoke.json`) only validate the harness; write them to a
scratch directory, never here.

## GB200 validation: 2026-11-16 UTC

The node was reached from `~/repos/yolo` with `uv run yolo ssh yupei-devbox-1115c0bdbd3`.
The isolated checkout remains at `/tmp/tspfold-c92f5b03/repo` on the node. Compound `-c` commands need
an explicitly quoted `bash -lc`; see `docs/GPU_RUNBOOK.md`.

| Item | Recorded environment |
| --- | --- |
| Devices | Physical GPUs 0 and 1 of four NVIDIA GB200 GPUs; only this pair was used |
| Memory | 189471 MiB per board in nvidia-smi; CUDA reports 197897617408 usable bytes |
| Interconnect | NV18; CUDA peer access and all FP32/BF16 collective/P2P checks pass |
| Hardware-gate bandwidth | 1 GiB messages: 670.93 GB/s one-way P2P, 625.34 GB/s per ring direction |
| Host / Python | aarch64 / 3.12.3 |
| PyTorch | `2.12.0a0+5aff3928d8.nv26.05` (NVIDIA container build, not the default lock) |
| CUDA / NCCL / driver | 13.2 / 2.30.4 / 580.159.04 |
| NumPy / Matplotlib / pytest | 2.1.0 / 3.10.9 / 9.2.0 |
| Source | Base `0fa880e` plus `env/source.patch`; uncommitted changes are explicitly recorded |

`env/env_manifest.json`, `env/runtime_packages.json`, and `env/hw_gate.json` contain the full
environment and hardware evidence. `env/source_sha256.txt` fingerprints the runtime modules. Reuse
the container stack through the isolated system-site-packages venv; do not apply the x86_64 `uv.lock`
or install over the shared system packages. Only missing pytest and its dependencies were added to
the private venv.

### Correctness and fixes

The final source passes **111 CPU tests**, **277 two-GPU records** (92 cases plus a ring check), and
**12 one-GPU smoke records**, with no tolerance changes. Both distributed methods pass reference,
repeatability, TP cross-rank equality, and causal future-perturbation checks. Maxima across the
two-GPU TP/TSP cases:

| Block | Dtype | Maximum absolute error | Maximum normalized L2 error |
| --- | --- | ---: | ---: |
| Attention | FP32, TF32 off | 2.864e-6 | 1.080e-6 |
| Attention | BF16 | 1.714e-2 | 4.190e-3 |
| Gated MLP | FP32, TF32 off | 5.175e-6 | 1.194e-6 |
| Gated MLP | BF16 | 2.004e-2 | 4.331e-3 |

The BF16 criterion is `abs(error) <= 0.02 + 0.02 * abs(reference)`, not an absolute-only 0.02 bound.
Every distributed result satisfies it. Per-method metrics, including unsharded precision context,
are in `correctness/summary.json` and the raw correctness JSONL files.

Issues reproduced and fixed on the node:

- The container's PyTorch predates `all_gather_single`; use the equivalent `all_gather_into_tensor`
  when the new API is unavailable.
- PCI bus IDs must retain their domain: these four GPUs share `01:00.0` in different domains.
  Strip ANSI styling from nvidia-smi topology headers, and return a nonzero exit status for a failed
  hardware gate.
- Gloo can report a peer-disconnection error before torchrun terminates the survivor of an injected
  single-rank OOM. The regression test now checks the actual OOM origin without rejecting that
  expected secondary error.
- Release TSP attention's `q` and `attn2d` views with their backing tensors. At B=1/S=8192, peak
  allocated memory fell from **216.125 to 184.125 MiB** (all heads), and **156.125 to 124.125 MiB**
  (bucket size 2). The before/after output checksums are identical.
- Exclude user annotations that duplicate NCCL kernel time; classify the observed GB200 `nvjet`
  matrix-multiply kernels as GEMMs.

`env/memory_lifetime_check.json`, `correctness/memory_{before,after}.jsonl`, and
`env/source_before_memory.patch` preserve the memory-fix evidence. These short smoke runs are
memory/checksum checks, **not** evidence of a timing speedup; their old profiler annotations must not
be mixed into the final kernel-time analysis.

### Performance and memory

**132/132 launches succeeded:** 96 main-sweep and 36 ablation records, with no OOMs, unsupported
cases, timeouts, or allocator retries. `analysis/validation.json` records the completeness checks.
Each configuration has three independent launches, 10 warmups and 30 measured iterations per launch.
Latency is the median of the three trial medians, with each iteration taking the maximum over the
two ranks. The model is h=2048, H=16, head dimension 128, I=8192, BF16; attention has 16,777,216
parameters and MLP has 50,331,648 parameters globally.

Selected main-sweep measurements at B=1, with TSP overlap enabled and all local heads in one bucket:

| Block | S | TP latency (ms) | TSP latency (ms) | TP peak allocated (MiB) | TSP peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention | 1024 | 0.331 | 1.247 | 40.125 | 61.571 |
| Attention | 8192 | 0.401 | 1.549 | 152.125 | 184.125 |
| Attention | 32768 | 2.039 | 8.310 | 536.125 | 616.125 |
| Gated MLP | 1024 | 0.290 | 0.496 | 80.125 | 116.125 |
| Gated MLP | 8192 | 0.628 | 0.650 | 248.125 | 200.125 |
| Gated MLP | 32768 | 1.959 | 1.840 | 824.125 | 488.125 |

Memory is **per GPU, taking the maximum across ranks**, not the sum over both GPUs. It includes
inputs, persistent local weights, outputs, and operator temporaries. Allocator peak reserved memory
and whole-device usage after measurement (including CUDA/NCCL overhead) are separate fields in the
raw records; whole-device usage is not a measured high-water mark.

The measured tradeoffs:

- **MLP:** the extra weight shard hurts short sequences. Memory breaks even at S=4096, falls 19.3%
  below TP at S=8192, and falls 40.8% below TP at S=32768. At the latter point TSP has 1.065x TP
  throughput. B=4/S=8192 gives a similar 1.062x throughput ratio and 40.8% memory reduction.
- **Default attention:** TSP is slower and uses more peak allocation at every main-sweep point.
  At S=32768 it is 4.08x slower with 14.9% more allocation. Lower per-rank input memory does not
  compensate for weight-transfer and gathered/reordered K/V temporaries.
- **Head bucketing:** in the S=32768 ablation, reducing the bucket from all 8 local heads to 1 cuts
  TSP peak allocation from 616.125 to 342.376 MiB (44.4%), but raises latency from 8.319 to
  10.558 ms (26.9%). Bucket size 4 gives 456.125 MiB and 8.952 ms.
- **Overlap:** at S=32768 it reduces MLP latency from 1.912 to 1.846 ms (3.4%); attention is
  essentially unchanged (8.322 versus 8.319 ms). The short-sequence MLP ablation is noisy and does
  not establish an overlap benefit.

The attention comparison uses identical `sdpa` selection policy but **different actual kernels**:
TP uses cuDNN square causal attention; TSP rank 0 mixes cuDNN and PyTorch FlashAttention, while rank 1
uses PyTorch FlashAttention for its lower-right non-square chunks. At S=32768, rank 0's profiler
attention-kernel totals are about 1.30 ms for TP and 5.97 ms for TSP (rank 1: 1.24 and 6.89 ms).
These totals are diagnostic, not additive to wall time. The slowdown is a result for this eager
implementation and backend dispatch, not evidence that every TSP implementation must be slower.

Seven of the 44 configurations have trial CV above 10% (maximum 21.5%), mostly at short sequence
lengths. Trial min/max and CV are retained in `analysis/summary.csv`, and plots show trial ranges.
Do not interpret small differences in these noisy configurations as reliable speedups.
TP/TSP MLP output checksums match exactly; the largest attention checksum relative difference is
5.13e-4. Checksums are a benchmark-scale sanity check, not a substitute for the reference-based
correctness suite above.

Complete results: `analysis/matched.csv` (16 matched pairs), `analysis/summary.csv` (44 configurations),
`analysis/summary.md`, and the `main_bf16_*` / `ablation_bf16_*` PNG/PDF plots. All four GPUs were idle
and nvidia-smi reported 0 MiB used after the runs (`env/gpus_after.csv`).

### Reusing the node checkout

```bash
# On the local machine:
cd ~/repos/yolo
uv run yolo ssh yupei-devbox-1115c0bdbd3

# Inside the node:
cd /tmp/tspfold-c92f5b03/repo
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
make test
make hw-check
make correctness-gpu
make sweep-main       # resumes existing records rather than rerunning them
make sweep-ablation
make analyze
```

The memory-limit sweep, final two-page report/PDF, and AI transcript exports were not part of this
validation run. No maximum feasible sequence length or end-to-end training/decoding result is claimed.
