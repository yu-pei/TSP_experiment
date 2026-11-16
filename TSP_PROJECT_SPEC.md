# Tensor Sequence Parallelism: Implementation and Evaluation Specification

## 1. Objective

Implement the forward-pass tensor and sequence parallelism (TSP) schedules in Algorithms 1 and 2 of **Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference**, by Vasu Shyam, Anna Golubeva, and Quentin Anthony. Establish numerical correctness, benchmark against standard tensor parallelism on two NVLink-connected GPUs, and document findings in a report of no more than two pages.

Reference: https://arxiv.org/abs/2604.26294; algorithm reference: https://arxiv.org/html/2604.26294v1. Record the paper version used.

This specification separates required deliverables from optional extensions. Implementation choices and benchmark defaults below are project decisions, not additional claims made by the paper.

## 2. Required scope

- TSP causal multi-head attention forward pass, following Algorithm 1.
- TSP gated MLP forward pass, following Algorithm 2, using SiLU/SwiGLU.
- Standard two-way tensor-parallel attention and gated MLP baselines.
- An unsharded reference for numerical validation on small configurations.
- Correctness tests and reproducible performance/memory experiments on exactly two NVLink-connected GPUs.
- A report of at most two pages covering approach, implementation, correctness evidence, measurements, and observations.
- An accurate AI-use declaration and associated chat logs if AI assists any part of the work.

Out of scope: backward passes, gradient validation, optimizer states, training convergence, full language-model training, autoregressive decoding/KV-cache serving, and reproduction of the paper's multi-node scale. Forward inference means processing an input sequence without gradients. A full decoder stack, GQA, positional encodings, and SP-only comparisons are optional extensions.

## 3. Implementation requirements

Use PyTorch distributed execution with one process per GPU and NCCL. Pin and record package versions. Favor readable modules and explicit shape/layout contracts; custom CUDA kernels are not required. Use a fused attention implementation where supported, and document actual backend selection for both methods.

### Shared contracts

Let B be batch size, S full sequence length, h hidden width, H attention heads, d=h/H head dimension, I the gated MLP intermediate width, and D=2. Require H and I divisible by D; require S divisible by 2D for equal zigzag chunks. Reject invalid configurations explicitly.

Inputs represent the same global tensor of shape [B,S,h] for every method. TP ranks receive the full sequence; TSP ranks receive their zigzag token subsets [B,S/D,h]. Weights represent identical global matrices and are partitioned identically along heads or intermediate channels. Use bias-free projections and zero dropout initially. Attention is causal. No approximation to the operator is permitted beyond floating-point rounding.

### TSP attention — Algorithm 1

1. Divide the full sequence into 2D equal contiguous chunks. Rank p owns chunks p and 2D-1-p.
2. Store one head shard of Q/K/V projections and its corresponding output-projection slice on each rank.
3. Iterate over weight-owning ranks in a consistent order. Broadcast a packed Q/K/V/output weight shard to all ranks.
4. Compute Q/K/V for the local tokens and current head shard.
5. Gather K/V along the sequence axis, optionally in configurable head buckets, and reorder gathered chunks into global token order.
6. Evaluate attention separately for the two local query chunks using their correct causal prefixes. Unequal query/key lengths require correct global-position or bottom-right causal alignment; do not assume a generic causal flag provides it.
7. Project the attended values with the matching output-weight slice and accumulate into the local output.
8. Return local output in the declared zigzag layout. Do not gather final outputs within the timed operation.

### TSP gated MLP — Algorithm 2

1. Shard gate/up projections along intermediate channels and the down projection along its corresponding input dimension.
2. For local tokens, compute SiLU(X W_gate^T) * (X W_up^T), apply the matching down projection, and accumulate locally.
3. Circulate weight shards using point-to-point send/receive. With two ranks, each applies its own shard and the peer's shard.
4. Launch the next transfer before current-shard computation where practical. Wait before consuming received data and preserve send-buffer lifetimes.
5. Return the complete local-token output without an activation all-reduce.

Only local persistent weight shards and bounded temporary shard buffers may reside on a GPU during a benchmark. Retaining a full unsharded model on each GPU invalidates the intended memory comparison. Measure buffers rather than assuming theoretical memory scaling.

## 4. Baseline choice and reasoning

**Required performance baseline: conventional TP with degree 2.**

| Baseline component | Implementation |
| --- | --- |
| Attention | Replicate input tokens; shard Q/K/V by heads; compute full-sequence causal attention for local heads; apply the row-parallel output projection; all-reduce partial outputs. |
| Gated MLP | Replicate input tokens; column-shard gate/up projections; row-shard down projection; all-reduce partial outputs. |

TP is a suitable primary baseline because it is standard, uses exactly the same two GPUs, and has the same persistent parameter-sharding degree as TSP. It provides a direct comparison between communicating activation outputs in TP and circulating/broadcasting weights plus exchanging K/V in TSP. Both methods perform the same global operator on the same batch and sequence, avoiding a comparison confounded by different model-replica counts.

A nontrivial orthogonal TP+SP mesh requires at least 2x2=4 GPUs. On two GPUs, choosing both dimensions larger than one is impossible. Therefore a four-GPU TP+SP baseline is outside the hardware requirement; using it as the main comparison would change the resource budget.

The unsharded reference is a correctness oracle, not the primary performance baseline. It establishes that both distributed methods compute the intended operation, but uses a different GPU count and memory layout.

An SP-only baseline is optional. It would clarify how much weight movement TSP adds relative to a sequence-sharded method with replicated weights. It is not necessary to satisfy the required comparison.

Fairness caveat: real TP implementations shard some intermediate activations even though inputs and outputs can be replicated. Do not assume TP retains every activation unsharded, or claim a guaranteed 2x memory reduction for forward-only TSP. Training-memory formulas from the paper do not directly describe this experiment.

## 5. Correctness protocol

- Generate deterministic global inputs and weights for small tests; distribute their exact shards to both methods.
- Compare TP and TSP separately against the unsharded reference for attention and MLP.
- Gather and undo the TSP zigzag layout only in the correctness harness, outside performance measurements.
- Cover D=1 smoke tests and required D=2 distributed tests; B=1 and B>1; multiple valid sequence lengths and head counts; head bucketing; and more than one random seed.
- Specifically test causal chunk boundaries, output-weight orientation, ring peer exchange, and sequence reconstruction. Perturb future tokens and verify earlier outputs remain unaffected.
- Start with FP32 and TF32 disabled for diagnostic tests, then validate the selected benchmark precision, preferably BF16 when supported.
- Initial acceptance thresholds: FP32 atol=1e-5, rtol=1e-4; BF16 atol=2e-2, rtol=2e-2 for standardized test inputs. These are proposed starting criteria. Investigate failures; justify and document any change before performance comparisons.
- Report maximum absolute error and normalized L2 error, norm(Y-Y_ref)/max(norm(Y_ref),epsilon), alongside allclose results. Do not rely solely on unstable elementwise relative errors near zero.
- Repeated calls must produce consistent outputs without hangs or buffer-lifetime errors.

## 6. Benchmark protocol

### Hardware gate

Record GPU models, memory capacities, driver, CUDA, PyTorch, NCCL, and attention-library versions. Capture nvidia-smi topology output, verify the GPU pair has an NVLink path and peer access, and run a small collective/P2P sanity check. Do not label a PCIe-only run as satisfying the hardware requirement. Hardware access is an execution prerequisite, not an assumption of availability.

### Workloads

Benchmark attention and gated MLP separately. A composed attention-plus-MLP forward pass is optional and must be labeled as such.

Proposed starting configuration: h=2048, H=16, d=128, I=8192; B in {1,2,4}; S in {1024,2048,4096,8192,16384,32768}. Use the same configuration for both methods. Extend S toward the observed memory limit when feasible. If hardware capacity demands a smaller model, choose it once, document it, and apply it to both methods. Report actual parameter counts rather than an approximate model-size label.

Use a compact sequence sweep at B=1 plus a batch sweep at a fixed feasible S. Avoid spending compute on an unnecessarily large Cartesian sweep. Record unsupported cases, OOMs, and runtime errors as distinct outcomes.

### Timing and memory

- Use inference mode/no gradients. Exclude initialization, data generation, correctness checks, and final-output gathering from timing.
- Include all operator-required broadcasts, ring transfers, K/V exchanges, and reductions.
- Use the same precision, mathematical operation, attention-backend policy, and global B/S across methods. Disclose unavoidable kernel differences.
- Start with 10 warmups and 30 measured iterations per case, repeated in 3 independent trials. Increase only if instability warrants it.
- Use synchronized wall-clock timing that waits for all work on all relevant streams; aggregate the maximum elapsed time across ranks. Keep reporting collectives outside the timed region.
- Define global throughput as B*S divided by distributed operation latency. Do not multiply by D: the two ranks process one global workload.
- Record median latency and trial variability. Keep the aggregation rule identical across methods.
- Measure steady-state peak allocated and reserved GPU memory after warmup, with peak statistics reset and consistent output lifetimes. Report the maximum across ranks and the pre-operation allocated baseline. Distinguish allocator memory from total device usage.
- Run each method/configuration in a fresh process group or otherwise demonstrate memory isolation. Never leave a correctness reference or the other method's parameters resident on the benchmark GPU.
- Bound run time and isolate failures so one rank's OOM does not leave another rank indefinitely blocked in a collective.

Save raw records with method, block, B/S/h/H/I, dtype, head-bucket size, overlap mode, hardware/software identifiers, seed, trial, latency, throughput, memory, and status/error. Preserve the exact launch commands and source revision.

### Analysis

Calculate speedup as TP latency / TSP latency and memory reduction as 1 - TSP peak allocated / TP peak allocated for matched configurations. Explain short-versus-long-sequence behavior using measured operator costs and communication sizes. Separate measured results from theoretical estimates. An optional overlap-on/off ablation can identify whether weight transfer is hidden. Do not infer end-to-end training or decoding performance from these block-level forward measurements.

## 7. Task backlog

All tasks below begin as not started. Completion is based on the listed evidence.

| ID | Task | Dependencies | Completion evidence |
| --- | --- | --- | --- |
| T01 | Freeze operator shapes, layouts, precision, and paper mapping | None | Configuration schema, documented exclusions, algorithm-to-module mapping. |
| T02 | Establish two-GPU runtime and verify NVLink | None | Environment manifest, topology capture, passing collective/P2P smoke test. |
| T03 | Implement unsharded attention and SwiGLU references | T01 | Deterministic reference outputs and basic causal tests. |
| T04 | Implement weight partitioning and zigzag helpers | T01, T03 | Round-trip sequence and weight reconstruction tests; invalid shapes rejected. |
| T05 | Implement TP attention and MLP baseline | T02, T03, T04 | Both blocks match reference on two ranks. |
| T06 | Implement TSP ring MLP | T02, T03, T04 | Two-rank reference agreement; bounded buffers; no forward output all-reduce. |
| T07 | Implement TSP attention | T02, T03, T04 | Packed broadcasts, K/V reorder, correct causal prefixes, head-bucket tests pass. |
| T08 | Complete distributed correctness suite | T05, T06, T07 | Error metrics across seeds/shapes/dtypes; repeated runs complete without hangs. |
| T09 | Implement benchmark runner and result schema | T01, T02 | Synchronized timing, memory isolation, raw results, failure/timeout handling. |
| T10 | Review overlap and memory behavior | T06, T07, T08, T09 | Verified stream/buffer dependencies; retained tensors inspected; optimized code retested. |
| T11 | Execute matched workload sweeps | T08, T09, T10 | Repeated two-GPU measurements and logged OOM/unsupported cases. |
| T12 | Analyze results and produce plots | T11 | Latency/throughput and memory plots, matched ratios, documented limitations. |
| T13 | Write report, maximum two pages | T12 | Approach, baseline rationale, correctness, results, observations, AI disclosure; page count checked. |
| T14 | Package reproducible deliverables | T08, T11, T13 | Clean-run instructions, pinned environment, source, configs, raw results, report, and AI chat logs. |

T03–T08 establish correctness before optimization claims. Benchmark-harness development can proceed before the final TSP implementation, but comparative results are accepted only after T08 passes. T02 blocks genuine performance validation if suitable hardware is unavailable.

## 8. Deliverables and completion criteria

Deliver a source project containing implementation modules, tests, environment/dependency instructions, benchmark configurations and commands, raw results, plot-generation code, and figures. Include a PDF report of at most two pages and an AI-use record with associated chat logs.

Suggested report allocation: approximately half a page for method and baseline, a quarter page for correctness and environment, one page for compact plots/results, and a quarter page for limitations and AI disclosure. Adjust the allocation while retaining all required topics. Keep full logs and detailed raw measurements in accompanying files.

The report must state the actual role of AI, including assistance with literature interpretation, specification, code, debugging, or writing as applicable. This project's specification and preceding paper discussion were AI-assisted and must be reflected accurately if used. Preserve the relevant conversation history now and continue recording later AI sessions; do not invent unavailable logs or claim the implementation was independently authored. Review logs for credentials and private information, marking any necessary redactions.

Completion requires both TSP blocks and the TP baseline to pass correctness checks, valid two-GPU NVLink measurements, reproducible artifacts, and a complete report within the page limit. A TSP speedup is not an acceptance requirement: slower execution or small memory savings are valid findings when measured correctly and explained. No benchmark numbers may be fabricated or presented as measured before execution.
