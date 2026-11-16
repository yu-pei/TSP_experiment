# Native-cuDNN head bucketing: memory savings with a latency cost

Measured on 2026-11-16 UTC, using the same two NV18-connected GB200 GPUs as the preceding backend
comparison. This experiment uses native `cudnn_frontend` attention throughout and freshly measures
TP, unbucketed TSP, TSP with head buckets 4/2/1, and MLP controls on the same pair.

**Memory savings hold for both attention and MLP. They do not imply a performance win for both.**
Four-head attention buckets offer the smallest latency penalty among the tested memory-saving
variants. One-head buckets save the most memory but are substantially slower.

## B=1, S=32768

BF16, h=2048, H=16, head dimension 128, I=8192, D=2. TSP weight-prefetch overlap is enabled.
Latency is the median of three independent trial medians; each trial has 10 warmups and 30 measured
iterations, taking the maximum over ranks. Memory is peak allocated per GPU, maximum over ranks
and trials, not whole-device usage.

| Block / variant | Latency (ms) | Peak allocated (MiB) | Memory saved vs TP | Latency / TP |
| --- | ---: | ---: | ---: | ---: |
| Attention TP | 2.062 | 536.125 | - | 1.00 |
| Attention TSP, all 8 local heads | 3.131 | 616.125 | -14.9% | 1.52 |
| Attention TSP, bucket 4 | 3.406 | 456.125 | **14.9%** | **1.65** |
| Attention TSP, bucket 2 | 5.175 | 376.125 | **29.8%** | 2.51 |
| Attention TSP, bucket 1 | 9.818 | 336.125 | **37.3%** | 4.76 |
| MLP TP | 2.001 | 824.125 | - | 1.00 |
| MLP TSP | 1.923 | 488.125 | **40.8%** | **0.961** |

Compared with unbucketed TSP, bucket 4 saves 160 MiB and adds about 8.8% latency. Compared with TP,
it saves 80 MiB but is still 65% slower. The memory-saving configuration does not retain the
unbucketed configuration's latency, and no tested bucket achieves TP latency parity.

The MLP control retains its memory benefit and has about 1.041x TP throughput at this point.
MLP has no head buckets; its implementation and transfer schedule were not changed.

## Other measured points and variability

At B=1/S=8192, attention bucket sizes 4/2/1 use 144.125/124.125/114.125 MiB versus TP's 152.125 MiB,
saving 5.3%/18.4%/25.0%. Their median latencies are 2.199/3.592/6.588 ms versus TP's 0.439 ms.
This short-sequence case is overhead-sensitive and does not demonstrate a useful speedup.

At B=4/S=8192, attention has the same memory footprint as B=1/S=32768, but bucket 4 takes 2.166 ms
versus TP's 1.114 ms. TSP MLP uses 488.125 versus 824.125 MiB and takes 1.879 versus 2.000 ms.

At B=1/S=32768, bucket-4 trial medians span 3.374--3.711 ms (CV 4.3%), and bucket-1 spans
8.980--10.327 ms (CV 5.7%). Seven shorter-context configurations have trial CV above 10%
(maximum 21.6%). All min/max ranges and CVs are retained; do not overinterpret small timing
differences. Peak allocation is stable across trials for every tested configuration.

## Why memory decreases while latency increases

Each D=2 weight shard covers eight heads. Bucketing limits the packed local K/V, all-gather output,
and reordered global K/V tensors to a subset of those heads. At B=1/S=32768 these simultaneous
temporary buffers total 320/160/80/40 MiB for buckets 8/4/2/1. This accounts for the corresponding
160/80/40 MiB reductions between the measured peaks.

The QKV projection and the output attention buffer still cover all local heads. Smaller buckets
do not reduce the mathematical attention work or communication volume; they increase the number
of gathers, reorders, and attention launches. The implementation still lacks pipelined K/V
gather/compute overlap. This experiment measures that tradeoff; it does not add an overlap
optimization or establish what a fully optimized pipeline could achieve.

## Validation and artifacts

- **63/63 fresh launches succeeded:** 45 attention and 18 MLP, covering 21 configurations with
  three trials each. No OOMs, unsupported cases, timeouts, or allocator retries.
- **97/97 dedicated bucket correctness records passed:** 49 at D=2 and 48 at D=1, using the existing
  float64 oracle, two seeds, B=1/B=4, all/4/2/1-head buckets, repeated calls, TP cross-rank equality,
  and future-token perturbations. Original BF16 tolerances were unchanged.
- The benchmark-scale output checksums match TP exactly for every measured bucket and MLP case.
  Correctness is established by the oracle suite, not by checksums alone.
- Trial-zero profiles confirm native cuDNN attention kernels on both ranks for every bucket;
  no fallback to PyTorch FlashAttention occurred.
- Runtime-module hashes match the preceding backend experiment; only a sweep config and Makefile
  target were added. Source patch, hashes, software/hardware records, worker identity, and per-trial
  idle-GPU snapshots are in `env/`. Data was checkpointed locally after each trial.

Primary result files:

| File | Content |
| --- | --- |
| `analysis/bucket_comparison.csv` / `.md` | Every bucket matched against same-run TP, plus MLP |
| `analysis/attention_bucket_tradeoff.png` / `.pdf` | Memory/latency tradeoff with trial ranges |
| `analysis/summary.csv` | All 21 configurations, variability, reserved/whole-device memory, kernel evidence |
| `analysis/validation.json` | Completeness, source, backend, and correctness audit |
| `cudnn_bucketing_bf16/raw/` and `logs/` | Raw measurements and per-launch logs |
| `correctness/` | Dedicated bucket correctness records, logs, and summary |

The standard analyzer's `matched.csv` and default sweep plots compare only unbucketed TSP against
TP; use `bucket_comparison.csv` and the tradeoff plot for the bucketed variants. Exact one-off
correctness/analysis producer source and commands are archived in `env/*_driver.json`.

## Reproduce

From the node checkout `/tmp/tspfold-c92f5b03/kernel-dispatch`:

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
make sweep-cudnn-bucketing RESULTS=results/native_bucketing
make analyze RESULTS=results/native_bucketing
```

The target resumes existing records. Use a fresh results root for independent measurements.
Do not replace the shared ARM64 container runtime with the repository's x86_64 lock.
This remains a forward-block experiment, not a full-model training/decoding or paper-scale result.
