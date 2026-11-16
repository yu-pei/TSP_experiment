# Design: operator contracts, layouts, and paper mapping (task T01)

## Paper and version

- Shyam, Golubeva, Anthony, *Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer
  Training & Inference*, arXiv:2604.26294.
- Version used: **v1**, HTML rendering at <https://arxiv.org/html/2604.26294v1> (accessed 2026-11-15).
  Algorithm line numbers below refer to Algorithms 1 and 2 of that version.

## Scope

Implemented: forward-only (inference mode, no autograd) TSP attention (Algorithm 1) and TSP gated MLP
(Algorithm 2), the conventional 2-way TP baseline for both blocks, and an unsharded float64 reference.

Excluded (spec section 2): backward passes, gradients, optimizer states, training, full decoder stacks,
autoregressive decoding/KV cache, GQA, positional encodings, normalization/residuals, SP-only baseline,
multi-node scale. Projections are bias-free, dropout is zero, attention is causal MHA.

## Configuration schema (`src/tspfold/config.py`)

`BlockShape(batch=B, seq_len=S, hidden=h, n_heads=H, intermediate=I, world_size=D)`; derived values
`d = h/H`, `H_s = H/D`, `h/D`, `I/D`, `S_loc = S/D`, `S_c = S/(2D)`.
Rejected with `ConfigError`: non-positive sizes, `h % H`, `H % D`, `I % D`, `S % 2D`, and a TSP head
bucket that does not divide `H_s`. The same validation applies to TP so both methods run identical
configurations. Benchmark/run options: method `tp|tsp`, block `attn|mlp`, dtype `fp32|bf16|fp16|fp64`,
`head_bucket` (TSP attention), `overlap` (TSP), attention backend (see below), seed.

## Layout contracts

| Tensor | TP (rank p) | TSP (rank p) |
| --- | --- | --- |
| Input | full `[B, S, h]` (replicated) | zigzag shard `[B, S/D, h]` = chunks `p` and `2D-1-p` |
| Output | full `[B, S, h]` (replicated after all-reduce) | zigzag shard `[B, S/D, h]`, same order as input |
| Attention weights | packed `[W_Q^(p); W_K^(p); W_V^(p); W_O^(p)T]`, `[4h/D, h]` | same buffer |
| MLP weights | packed `[W_gate^(p); W_up^(p); W_down^(p)T]`, `[3I/D, h]` | same buffer |

- Global weights use the `nn.Linear` convention `W[out, in]`, `y = x W^T`.
- Head shard `p` = heads `p*H/D ... (p+1)*H/D - 1` = rows `p*h/D ... (p+1)*h/D - 1` of `W_Q, W_K, W_V`
  and the same *columns* of `W_O`, stored transposed (`W_O^(p)T`, as in Algorithm 1 line 8).
- MLP shard `p` = intermediate channels `p*I/D ...` of `W_gate, W_up` (rows) and of `W_down`
  (columns, stored transposed as `W_2^(p)` in Algorithm 2).
- Packing makes every transfer one contiguous buffer (one broadcast / one send per shard) and makes
  persistent memory identical for TP and TSP.
- Zigzag (`layout.py`): the sequence is cut into `2D` chunks of `S_c` tokens; rank `p` stores
  `[chunk_p ; chunk_{2D-1-p}]`. Gathering and un-zigzagging outputs happens only in the correctness
  harness, never in timed code.
- Inputs: one deterministic global tensor `X ~ N(0,1)` defined per 256-token block
  (`weights.make_input_range`), so TSP ranks can generate exactly their slices of it. Weights:
  `N(0, 1/fan_in)`, generated on CPU, sharded on CPU, and only the local shard is moved to the GPU.

## Algorithm 1 -> `TSPAttention` (`src/tspfold/tsp.py`)

| Alg. 1 line | Implementation |
| --- | --- |
| 1-3 | `TSPAttention(packed, shape, head_bucket, overlap, backend)`; input `[B, S/D, h]` zigzag shard |
| 5, 12 | `S_c = shape.chunk_len`; `(s1, s2) = layout.causal_key_bounds(p, D, S)` = `((p+1)S_c, (2D-p)S_c)` |
| 7 | `for r in range(D)` over weight owners, same order on every rank |
| 8 | `dist.broadcast(buf, src=r)` of the packed `[4h/D, h]` shard; owner broadcasts its persistent buffer, others receive into a temporary buffer |
| 9 | unpack = views `w[:3h/D]` (`W_Q;W_K;W_V`) and `w[3h/D:]` (`W_O^(r)T`), no copy |
| 10-11 | one fused GEMM `X [W_Q;W_K;W_V]^T` viewed as `[B, S_loc, 3, H_s, d]` (equivalent to three products) |
| 13 | `attn = empty([B, S_loc, H_s, d])` (every element is written below) |
| 14-15 | loop over head buckets; local K/V of the bucket packed into one `[2, B, S_loc, H_b, d]` buffer |
| 16 | `all_gather_stacked` (one collective for K and V) -> `[D, 2, B, S_loc, H_b, d]`; `all_gather_single` on newer PyTorch, `all_gather_into_tensor` before the API rename |
| 17 | `layout.zigzag_reorder_gathered`: one concatenation kernel -> global order `[2, B, S, H_b, d]` |
| 18-20 | query chunks `q[:, :S_c]`, `q[:, S_c:]` attend to key prefixes `[:s1]`, `[:s2]` with **bottom-right** causal alignment (`ops.causal_attention`) |
| 21 | results written into `attn[:, :S_c, bucket]` / `attn[:, S_c:, bucket]` |
| 23 | `Y = A W_O^(r)T` for the first owner, then `Y.addmm_(A, W_O^(r)T)` |
| 25 | return `Y` in the input's zigzag layout; no output gather |

`overlap=True` (default) prefetches broadcast `r+1` before computing shard `r`; the owner does not
stall on its own broadcast (it only reads its persistent buffer) and waits on that handle before
returning. `overlap=False` issues each broadcast synchronously, literally as written.
The `q` and `attn2d` views are explicitly released with their backing tensors after each owner, so
previous-owner QKV/attention storage is not retained while the next owner's K/V buffers are allocated.

## Algorithm 2 -> `TSPMLP` (`src/tspfold/tsp.py`)

| Alg. 2 line | Implementation |
| --- | --- |
| 1-6 | `TSPMLP(packed, shape, overlap)`; `current = packed` (persistent local shard) |
| 7 | `for t in range(D)` |
| 8-10 | `ring_exchange(current, incoming)`: `batch_isend_irecv` send to `p+1`, receive from `p-1` into a fresh temporary buffer, issued **before** the shard's GEMMs |
| 12-14 | `ops.swiglu_shard`: one GEMM for `[W_gate; W_up]`, in-place `SiLU(g) * u`, `Y.addmm_(., W_down^(r)T)` |
| 15-17 | wait on both handles before the received shard is consumed; the previous send buffer is released only after its send completed |
| 20 | return local `Y`; **no activation all-reduce** |

`overlap=False` waits for each transfer before computing (ablation). With `D=2`, each rank applies its
own shard and then its peer's shard; at most one temporary shard buffer is alive.

## TP baseline (`src/tspfold/tp.py`)

Attention: replicated `[B, S, h]` input; fused QKV GEMM for the local heads; square causal attention
over the full sequence; row-parallel output projection `A W_O^(p)T`; `all_reduce(SUM)`.
MLP: replicated input; `ops.swiglu_shard` on the local channels (same kernel sequence as TSP);
`all_reduce(SUM)`. Both return the replicated full output.

## Attention kernel policy (`src/tspfold/ops.py`)

`causal_attention(q, k, v)` takes `[B, L, H, d]` and enforces bottom-right alignment: query `i` of an
`L_q`-long chunk attends to keys `j <= i + (L_k - L_q)`.
- Square inputs (all of TP; TSP rank 0's first chunk): `F.scaled_dot_product_attention(is_causal=True)`.
- Non-square inputs: `attn_mask=torch.nn.attention.bias.causal_lower_right(L_q, L_k)`. In torch 2.14
  this dispatches to the FlashAttention kernel (whose causal flag is lower-right) or to the
  mem-efficient kernel's lower-right mask type, otherwise to a materialized mask. A plain
  `is_causal=True` is **top-left** aligned for `L_q != L_k`; `tests/test_ops.py` guards this pitfall.
- Backend policy is a run option applied identically to both methods for the whole forward:
  `sdpa` (PyTorch's choice, default), `flash`, `efficient`, `math`, `cudnn` (SDPA restricted via
  `sdpa_kernel`), `flash_attn` (optional Dao-AILab package), `naive` (explicit). The kernels that
  actually ran are recorded per method by the benchmark's profiler pass (`profile.attention_kernels`).
- `cudnn_frontend` is an explicit, optional alternative using cuDNN graph SDPA with
  `diagonal_alignment=BOTTOM_RIGHT` and `diagonal_band_right_bound=0`, for square and non-square
  CUDA FP16/BF16 inputs. It does not construct an S-by-S mask or pad the queries. The tested NVIDIA
  PyTorch 2.12 `CausalBias` dispatch skips cuDNN for non-square lower-right attention, so this path
  bypasses that dispatch limitation without changing `sdpa` or the operator.

The native cuDNN path is forward-only and rejects enabled autograd on gradient-requiring inputs.
Plans are cached by dtype, shape, and all input strides, with at most 32 plans per thread/device.
The cache holds only graph metadata, not inputs, outputs, or GPU workspace tensors. Each invocation
binds fresh tensor pointers and uses the caller's current CUDA stream; output and workspace lifetimes
remain visible to the allocator measurements. Planning occurs on the first warmup, outside timing.
This changes kernel dispatch only; it does not implement the paper's staggered K/V/weight pipeline.

## Precision policy

- Diagnostics: FP32 with TF32 disabled (`torch.backends.cuda.matmul.fp32_precision = "ieee"`).
- Benchmark: BF16 weights and activations; FP32 accumulation inside GEMM/attention kernels;
  partial outputs are accumulated in BF16 by both methods (TP: BF16 all-reduce; TSP: BF16 `addmm_`).
- Oracle: float64 unsharded reference computed from the identical dtype-rounded inputs and weights.
  The correctness records also include an "unsharded" single-device run with the same kernels and
  dtype, to show the error level expected from the precision alone.

## Measurement methodology (summary; details in `src/tspfold/bench.py`)

- One fresh `torchrun` process group per case (memory isolation); 10 warmups, 30 measured iterations,
  3 independent launches (trials) with TP/TSP order alternating per trial. Warmups use exactly the
  measured-iteration pattern below, so the allocator cache reflects synchronized execution.
- Iteration timing: barrier (untimed) -> device synchronize -> forward -> device synchronize, so all
  compute and NCCL streams are included; per-iteration latency = max over ranks; trial latency =
  median; configuration latency = median of trial medians (min/max reported).
- Memory: peak allocator stats reset after warmup, read after the measured iterations; outputs are
  freed every iteration; pre-operation allocated baseline, peak allocated/reserved (max over ranks),
  and whole-device usage (`cudaMemGetInfo`, includes CUDA context and NCCL buffers) are recorded.
- Reporting collectives, output checksums, and the optional profiler pass run after measurement.
- Profiler categories count device kernels, not user annotations that duplicate NCCL durations.
  GB200 `nvjet` matrix-multiply kernels are classified as GEMMs. Actual attention backends are retained
  separately for each rank; an identical `sdpa` policy can choose different square/non-square kernels.
- Failures: per-rank status files; the failing rank exits without entering collectives, torchrun
  tears down its peer; the sweep driver enforces a wall-clock timeout and records `oom`, `unsupported`,
  `invalid_config`, `error`, `crash`, `timeout`, or `skipped` as distinct outcomes.

## Known caveats

- The per-iteration barrier aligns rank start times only to within the barrier's completion skew
  (microseconds); this bias is identical for both methods.
- NCCL internal buffers are outside the PyTorch allocator; they appear only in whole-device usage.
- Profiler kernel time for NCCL includes time spent waiting for the peer and overlaps with compute,
  so category times are not additive to wall-clock latency.
- The paper's training-memory formulas do not describe this forward-only experiment; TP shards some
  intermediates too, so no fixed memory ratio is assumed.
