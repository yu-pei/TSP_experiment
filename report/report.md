# Forward TSP versus TP: memory and performance

<!-- Print-checked at two pages on A4 and US Letter: 11 pt, 1.15 line spacing, 0.65-inch margins. -->

**Experiments:** November 16, 2026 UTC. **Reference:** Shyam, Golubeva, and Anthony,
*Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference*,
arXiv:2604.26294 v1.

## Approach

We implemented the paper's forward attention and gated-MLP schedules and compared them with
conventional two-way tensor parallelism (TP) on two NVLink-connected GPUs, using identical inputs,
weights, precision, and workloads. We evaluate forward blocks only: backward/training, decoding,
and a four-GPU TP+SP baseline are excluded. This is not a paper-scale reproduction.

TSP partitions tokens into four chunks, assigning each rank a symmetric pair in zigzag order.
For attention, ranks broadcast packed head-weight shards, project local tokens, gather and reorder
K/V, evaluate each query chunk against its causal prefix, and accumulate output projections.
For the SwiGLU MLP, ranks circulate intermediate-channel weight shards and accumulate local-token
outputs. TP instead replicates input tokens and all-reduces partial outputs. Neither method gathers
TSP's final outputs inside the measured operation.

## Implementation and investigation

The implementation uses PyTorch distributed/NCCL, fused QKV and gate/up projections, and bounded
temporary weight storage. Attention uses **bottom-right causal alignment** for unequal query/key
lengths. Head bucketing exchanges and evaluates subsets of heads, reducing temporary K/V storage
without approximating attention. Weight transfers are prefetched, but K/V gather/compute is not
fully pipelined.

GPU validation fixed collective-API compatibility, topology parsing, retained tensor views, and
duplicate profiler accounting. Controlled investigation found that PyTorch's lower-right dispatch
bypassed cuDNN. An explicit
`cudnn_frontend` backend now uses native bottom-right masking, stride-aware metadata-only plan
caching, and the caller's CUDA stream. The original default remains reproducible.

In separate same-worker controls at B=1/S=32768, this backend change reduced unbucketed TSP attention from
**8.412 to 3.019 ms (2.79x)**, with unchanged allocation. Forcing TP onto FlashAttention also
slowed it, identifying dispatch mismatch as the main cause of the original kernel-time gap.

## Correctness evidence

The 120-test regression suite passed. GPU suites produced **289 standard, 133 native-cuDNN, and
97 dedicated bucketing records**, all passing; these include unsharded precision-context records.
Checks cover D=1/2, multiple shapes and seeds, B=1/2/4, head buckets, repeated calls, ring exchange,
TP cross-rank equality, and perturbing future tokens without changing earlier outputs.

TP and reconstructed TSP outputs are compared with an unsharded float64 oracle outside benchmarks.
Tolerances were unchanged: FP32 atol/rtol = 1e-5/1e-4 with TF32 disabled; BF16 = 0.02/0.02.
The standard distributed FP32 maximum absolute error was 5.18e-6. Dedicated bucket tests reached
2.18e-2 absolute and 4.19e-3 normalized L2 error; every element satisfied the combined
absolute-plus-relative criterion. Benchmark checksums also matched exactly across the final
bucket/MLP configurations, as a supplementary check.

## Benchmark results

Hardware: two GB200 GPUs, 189471 MiB each, NV18 and verified peer/collective access. Runtime:
ARM64 Python 3.12.3, NVIDIA PyTorch 2.12 development build, CUDA 13.2, cuDNN 9.22, NCCL 2.30.4,
driver 580.159.04. A private environment reused the container GPU stack. After a worker replacement,
all backend and bucket comparisons were rerun on the same replacement worker.

Each configuration uses three fresh launches, ten warmups and thirty measured iterations.
Synchronized wall time includes required communication; each iteration takes the maximum over
ranks, followed by median-over-iterations and median-over-trials. Memory peaks are reset after
warmup; initialization, oracle checks, output gathering, and profiling are excluded.
Initial sweeps varied S=1024--32768 at B=1 and B=1/2/4 at S=8192. Follow-ups measured
(B,S)=(1,8192), (1,32768), and (4,8192).

**Final same-run results:** B=1, S=32768, h=2048, H=16, head dimension 128, I=8192, BF16.
Attention uses native cuDNN. Memory is maximum **peak allocated per GPU**, not total device usage.

| Block / variant | Latency (ms) | Peak (MiB) | Memory saved vs TP |
| --- | ---: | ---: | ---: |
| Attention TP | 2.062 | 536.125 | - |
| Attention TSP, bucket 8 (all) | 3.131 | 616.125 | -14.9% |
| Attention TSP, bucket 4 | 3.406 | 456.125 | 14.9% |
| Attention TSP, bucket 2 | 5.175 | 376.125 | 29.8% |
| Attention TSP, bucket 1 | 9.818 | 336.125 | 37.3% |
| MLP TP | 2.001 | 824.125 | - |
| MLP TSP | 1.923 | 488.125 | 40.8% |

## Observations and limitations

**Both blocks save memory, but their performance tradeoffs differ.** Four-head attention buckets
provide the least latency penalty among tested memory-saving variants: 14.9% less allocation at
1.65x TP latency. One-head buckets save 37.3% but take 4.76x TP latency. MLP saves 40.8% with
1.041x TP throughput at this point.

At 32K tokens, simultaneous packed/gathered/reordered K/V buffers occupy 320/160/80/40 MiB for
buckets 8/4/2/1. Smaller buckets reduce these copies but increase communication and kernel-launch
counts. They do not reduce total attention work. Bucket-4 trial medians were 3.374--3.711 ms;
shorter-context configurations were noisier, with CV up to 21.6%. No tested attention bucket reached
TP latency parity. These results establish a memory benefit, not a universal speedup or validation
of the paper's broader training claims; the full overlap pipeline and memory-limit sweep remain undone.

## How I used AI

I used AI for specification, implementation, experiments, diagnosis, and reporting. I supplied goals
and GPU access, challenged the initial slowdown explanation as possible backend misconfiguration,
and requested the combined backend/bucketing experiment. Copilot designed and ran controls,
generated most code, tests, analysis, and this report, and recovered work after the node restart.
My prompts directed and challenged AI-generated work; they were not independent manual implementation.
Human review remains pending.

**Evidence:** `results/native_bucketing/` (63 launches), `results/attention_dispatch/` (54 controls),
and `results/` (132 original launches) retain raw measurements, source provenance, correctness, and
plots. AI ledger and transcript limitations: `docs/AI_USE.md`.
