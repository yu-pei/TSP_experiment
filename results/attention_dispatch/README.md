# Attention backend diagnosis and correction

Measured on 2026-11-16 UTC on two NV18-connected GB200 GPUs after the YOLO worker was replaced.
All comparisons here use the **same replacement worker**; the earlier main/ablation measurements in
the parent directory are preserved separately and are not used to calculate these speedups.

## Result

The large attention-kernel gap was primarily a backend-dispatch issue, not an unavoidable cost of
TSP's smaller query chunks. The installed NVIDIA PyTorch 2.12 `CausalBias` lower-right branch checks
FlashAttention and memory-efficient attention but does not select cuDNN. Consequently, the original
`sdpa` policy selected native SM100 cuDNN for TP's square attention, but PyTorch FlashAttention for
TSP's rectangular chunks.

The new explicit `cudnn_frontend` backend uses native cuDNN bottom-right diagonal masking, without
a dense mask, padded queries, changed scaling, or relaxed tolerances.

### B=1, S=32768, BF16

Model dimensions are h=2048, H=16, head dimension 128, I=8192, D=2. TSP uses all local heads in
one bucket and weight-prefetch overlap; those settings are identical across the three policies.

| Backend policy | TP latency (ms) | TSP latency (ms) | TP peak allocated (MiB) | TSP peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `sdpa` (original mixed dispatch) | 2.048 | 8.412 | 536.125 | 616.125 |
| `flash` (both methods restricted) | 6.623 | 8.509 | 536.125 | 616.125 |
| `cudnn_frontend` (both methods native cuDNN) | 2.142 | 3.019 | 536.125 | 616.125 |

TSP's forward latency improves **2.79x** over its `sdpa` control. Forcing TP onto Flash also makes
it substantially slower, isolating the backend effect from TSP's scheduling differences.
With native cuDNN on both sides, TSP is still about **41% slower than TP**; the full staggered
weight/K/V/compute pipeline has not been implemented.

The profiler confirms the kernel change. Per-iteration attention-kernel totals at S=32768:

| Method / policy | Rank 0 (ms) | Rank 1 (ms) |
| --- | ---: | ---: |
| TP / `sdpa` | 1.341 | 1.247 |
| TSP / `sdpa` | 5.967 | 6.962 |
| TP / `cudnn_frontend` | 1.364 | 1.413 |
| TSP / `cudnn_frontend` | 1.593 | 1.487 |

The native TSP profiles contain only cuDNN attention kernels, not `pytorch_flash` kernels. Category
times are not additive to wall time: NCCL time includes peer waiting and may overlap compute.
Communication, K/V staging/reordering, GEMMs, and host launch overhead remain in the measured forward.

### Other measured points

| B | S | TSP `sdpa` (ms) | TSP native cuDNN (ms) | TSP peak allocated, both policies (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8192 | 1.543 | 1.394 | 184.125 |
| 4 | 8192 | 3.267 | 1.959 | 616.125 |

The short B=1/S=8192 cases are noisy: native TSP's trial CV is 21.2%, with trial medians
1.373--2.117 ms. Do not treat its modest median improvement as a robust small-shape speedup.
The B=1/S=32768 native-TSP trial medians span 2.995--3.155 ms (CV 2.3%).

## Correctness, memory, and provenance

- The 120-test regression suite passed (112 CPU and 8 GPU cases). The focused 8-case GPU suite was
  also rerun after strengthening the cache test to check actual allocated storage, not only Python
  tensor lifetimes.
- Native-cuDNN block correctness passed 61 D=2 and 72 D=1 records, covering both TP and TSP, two seeds,
  batch sizes, head buckets, overlap settings, repeatability, and causal future-perturbation checks.
  The existing 289 standard GPU correctness records also passed on this worker.
- All 54 launches succeeded: three policies, two methods, three (B,S) points, and three independent
  trials. Each launch used 10 warmups and 30 measured iterations; trial-zero records include five
  additional profiling iterations outside timing/memory measurement.
- No allocator retries occurred. Peak allocated memory is unchanged by the backend replacement.
  Baseline allocation remains exactly input + persistent local weights + 8.125 MiB workspace.
  Plans cache metadata only; GPU outputs/workspaces are allocated per call on the caller's stream.
- `env/source.patch` includes newly added files as well as tracked-file changes. Applying it to
  base commit `0fa880e` reconstructs all runtime hashes in `env/source_sha256.txt`.
  The final test-only strengthening does not change any measured runtime-module hash.
- `env/run_context.json` records the worker replacement and identity hash. `env/hw_gate.json`,
  `env/runtime_packages.json`, and each raw record identify the hardware/software stack.
  Results were copied locally after each sweep to avoid losing completed data to another restart.

Raw records/logs and exact launch commands are under `attention_{cudnn,sdpa,flash}_bf16/`.
Generated tables, 12 PNG/PDF plots, and the completeness audit are in `analysis/`.
Detailed correctness metrics are in `correctness/`.

## Reproduce

Use the restored checkout `/tmp/tspfold-c92f5b03/kernel-dispatch` on the existing YOLO job.
The NVIDIA runtime already contains frontend 1.23.0; only missing pytest was restored in a private
system-site-packages venv. Do not run `uv sync` over the shared ARM64 container runtime.

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
make sweep-attention-cudnn
.venv/bin/python -m tspfold.sweep --config results/env/sweep_attention_sdpa.json --results-dir results
.venv/bin/python -m tspfold.sweep --config results/env/sweep_attention_flash.json --results-dir results
make analyze
```

In the local repository, the corresponding output root is `results/attention_dispatch`; pass
`RESULTS=results/attention_dispatch` to the Makefile targets, or `--results-dir results/attention_dispatch`
to the sweep runner. The original `sdpa` default is deliberately unchanged; select the new path with
`--attn-backend cudnn_frontend`.
