# AI-use record

Running declaration required by the specification (sections 2 and 8). This ledger covers all
**20 recorded human messages** in the current GPU-investigation session, including explanatory
questions, repeated requests, progress checks, report preparation, and the commit/push request, as well as the earlier
AI-assisted phases known from the original project record. Current-session dates/times below are
UTC (the user's local date was November 15, 2026).

The authentic user-facing transcript is
[`ai_logs/2026-11-16_gpu-investigation.md`](../ai_logs/2026-11-16_gpu-investigation.md).
It includes visible assistant replies and tool-delivered completion summaries, with marked privacy
redactions and an aggregate tool inventory. It is a snapshot through report delivery and the packaging request,
not a claim that earlier sessions or future messages have been exported. The earlier paper/specification
and initial implementation transcripts are **unavailable on this machine**; they have not been
reconstructed. Do not describe the implementation or report as independently authored.

## How the user used AI

The user set the goals, supplied GPU access, and directed an iterative investigation. In particular,
the user challenged the initial explanation of attention's slowdown and proposed a backend
misconfiguration, leading Copilot to design controls separating kernel selection from shape effects.
The user then requested the combined native-cuDNN/bucketing experiment and asked for explanations of the paper's
claims, memory accounting, batch/sequence scaling, overlap, and head bucketing.

Copilot generated most implementation code, tests, configurations, benchmark/analysis tooling, fixes,
documentation, and the Markdown report; executed the experiments; diagnosed failures; and restored
the experiment after a worker replacement. The user's questions changed the experimental direction
but are not evidence of independent manual implementation or a completed human code review.
Human review remains pending.

## Summary

| Phase | AI role | Human role | Chat log |
| --- | --- | --- | --- |
| Paper discussion and project specification (`TSP_PROJECT_SPEC.md`) | Literature interpretation and specification drafting, per the original record | Directed the discussion; adopted the specification | Earlier transcripts **unavailable here; not exported** |
| Repository creation and local implementation, 2026-11-15 | Copilot SDK in VS Code wrote essentially all code, tests, configs, scripts, documentation, and the LaTeX skeleton, per the original record | Requested local implementation and validation before moving to GPUs | Earlier transcript **unavailable here; not exported** |
| GPU validation and measurement, 2026-11-16 UTC | Copilot CLI checked access/environment, fixed portability/memory/profiler issues, and ran 132 baseline/ablation launches | Provided the node and requested validation and performance/memory measurements | Current transcript U01-U02 |
| Interpretation and experimental design | Explained paper-comparison limits, memory accounting, batch/sequence scaling, and missing overlap; revised conclusions as evidence accumulated | Asked the questions and challenged assumptions | Current transcript U03-U08, U12, U15 |
| Attention backend diagnosis and recovery | Implemented native cuDNN, restored the replacement worker, and ran 54 same-worker controls | Suggested misconfiguration, reported worker loss, and requested status | Current transcript U08-U11 |
| Native-cuDNN bucketing | Ran 97 dedicated correctness records and 63 attention/MLP launches; generated matched memory/latency analysis | Requested the combined experiment twice and asked about its mechanism/progress | Current transcript U13-U17 |
| Markdown report and AI audit | Wrote `report/report.md`, checked a two-page print rendering, exported available visible history, and expanded this ledger | Requested an at-most-two-page Markdown report and an explicit account of personal AI use | Current transcript U18-U19, including report delivery; PDF not requested |
| Commit and GitHub publication | Reviewed accumulated changes, checked the public remote, redacted internal network identifiers from NCCL logs, and prepared the commit/push | Authorized committing and pushing the work | Current transcript U20; export precedes the final packaging acknowledgement |

## Complete current-session interaction ledger

The U identifiers link to exact, redacted user messages in the exported transcript. Tool-generated
measurements, not conversational hypotheses, are the source of the report's numerical claims.

| Message / UTC time | User request | AI response or action |
| --- | --- | --- |
| [U01, 05:06](../ai_logs/2026-11-16_gpu-investigation.md#u01) | Validate the repository on the existing GPU node; measure performance/memory and fix issues | Checked access/runtime, established an isolated environment, fixed portability/topology/memory/profiler issues, and ran correctness plus main/ablation sweeps |
| [U02, 05:34](../ai_logs/2026-11-16_gpu-investigation.md#u02) | Current status | Reported actual progress; continued measurements and saved results |
| [U03, 05:56](../ai_logs/2026-11-16_gpu-investigation.md#u03) | Does this validate the paper? | Distinguished forward-block correctness and partial supporting evidence from paper-scale empirical reproduction |
| [U04, 06:04](../ai_logs/2026-11-16_gpu-investigation.md#u04) | Is extra attention memory communication-related? | Reconciled the 80 MiB excess with K/V staging and a temporary weight shard; separated allocator-managed buffers from hidden NCCL allocations |
| [U05, 06:11](../ai_logs/2026-11-16_gpu-investigation.md#u05) | Batch versus sequence-length sweeps | Explained the existing sweeps and the distinction between B*S storage/MLP scaling and sequence-dependent attention work |
| [U06, 06:20](../ai_logs/2026-11-16_gpu-investigation.md#u06) | Direct paper results for forward attention/MLP | Clarified that the cited paper's aggregate throughput is not a per-block ms/MiB target; those external figure values are not used in the report |
| [U07, 06:27](../ai_logs/2026-11-16_gpu-investigation.md#u07) | Is communication overlap missing? | Inspected the implementation: weight prefetch exists, but K/V gather/compute is not fully pipelined; identified a separate kernel-dispatch gap |
| [U08, 06:33](../ai_logs/2026-11-16_gpu-investigation.md#u08) | Suspected misconfigured attention kernels | Inspected installed dispatch, ran identical-backend/shape controls, implemented native lower-right cuDNN, and validated it |
| [U09, 06:48](../ai_logs/2026-11-16_gpu-investigation.md#u09) | Suspected node replacement | Confirmed a different host under the same job, restored local artifacts and the private environment, and reran hardware/correctness checks |
| [U10, 07:07](../ai_logs/2026-11-16_gpu-investigation.md#u10) | Current status | Reported completed native/SDPA controls and remaining Flash controls; did not present incomplete work as finished |
| [U11, 07:28](../ai_logs/2026-11-16_gpu-investigation.md#u11) | Current status | Reported completion: 120 regression tests, 422 GPU records, 54 comparison launches, and remaining attention overhead |
| [U12, 07:30](../ai_logs/2026-11-16_gpu-investigation.md#u12) | Combine the improved latency and bucketing memory claims? | Explicitly said the two gains had been measured separately and could not yet be claimed for one configuration |
| [U13, 07:34](../ai_logs/2026-11-16_gpu-investigation.md#u13) | Run native cuDNN with bucketing | Request retained; followed by the repeated request below |
| [U14, 07:37](../ai_logs/2026-11-16_gpu-investigation.md#u14) | Repeated bucketing request | Ran one combined experiment, not two: buckets all/4/2/1 plus same-worker MLP controls, with local checkpoints |
| [U15, 07:40](../ai_logs/2026-11-16_gpu-investigation.md#u15) | What is head bucketing? | Explained exact per-head K/V exchange/evaluation and its memory/launch-count tradeoff; clarified that QKV projection is still over all local heads |
| [U16, 07:43](../ai_logs/2026-11-16_gpu-investigation.md#u16) | When can memory/performance tradeoffs be assessed? | Gave a progress estimate and explicitly provisional first-trial figures, then completed the remaining trials |
| [U17, 07:50](../ai_logs/2026-11-16_gpu-investigation.md#u17) | Close to completion? | Reported 62/63 runs, finished auditing/analysis, and recorded memory savings with attention latency penalties |
| [U18, 07:56](../ai_logs/2026-11-16_gpu-investigation.md#u18) | At-most-two-page Markdown report and complete AI tracking | Wrote the evidence-based report, checked two-page A4/US Letter renderings, and audited/exported available history |
| [U19, 07:56](../ai_logs/2026-11-16_gpu-investigation.md#u19) | Summarize how the user used AI in the report | Added a first-person disclosure distinguishing user direction/challenges from AI-generated implementation, execution, analysis, and writing |
| [U20, 08:07](../ai_logs/2026-11-16_gpu-investigation.md#u20) | Commit and push to GitHub | Reviewed source/artifacts and remote state, checked for secrets, redacted internal hostname/IP fields in NCCL logs, and prepared publication; Git history records the resulting commit |

## Corrections and boundaries

- Initial comparisons used the same `sdpa` policy but different actual attention kernels. Later
  same-worker controls established a dispatch issue and reduced native TSP latency; original results
  remain intact rather than being relabeled.
- Kernel-dispatch improvements and memory savings were initially separate experiments. The combined
  native-cuDNN bucket run subsequently established both in one configuration, but not a TP-matching
  attention speedup: at 32K, bucket 4 saves 14.9% at 1.65x TP latency; bucket 1 saves 37.3% at 4.76x.
- Benchmark checksums supplement, not replace, oracle-based correctness. Record totals include
  unsharded precision-context records and are not counts of independent model-training runs.
- The final report uses local raw measurements and explicitly excludes full-model, training,
  decoding, fully pipelined overlap, and paper-scale reproduction claims. External-paper discussion
  in the historical transcript is not additional experimental evidence.

## Details of the 2026-11-15 implementation session

The following describes the earlier session according to the pre-existing AI-use record. Its
transcript is unavailable here; this is not a newly reconstructed or independently verified log.

What the assistant did:

- Read Algorithms 1 and 2 in the arXiv v1 HTML version and mapped them to modules
  ([DESIGN.md](DESIGN.md)).
- Chose and pinned the environment: Python 3.12, `torch==2.14.1` (cu126 build on Linux) via `uv.lock`.
- Implemented `src/tspfold/` (contracts, zigzag/weight helpers, float64 reference, TP baseline, TSP
  attention and ring MLP, correctness suite, benchmark runner, sweep driver, hardware gate, analysis,
  theoretical communication model), the tests, configs, Makefile, README, DESIGN, GPU runbook, this
  record, and `report/report.tex` (skeleton without results).
- Ran local CPU/gloo validation: unit tests; distributed correctness at D=1 and D=2 in FP32 and BF16;
  mutation checks (deliberately broken causal bounds, alignment, zigzag reorder, output-weight
  orientation, and ring buffer handling were each detected by the suite); and benchmark/sweep harness
  smoke runs including an injected single-rank OOM. CPU timings were used only to exercise the harness
  and are not results.
- Debugging during the session: `torchrun --standalone` hanging on macOS (switched to static
  rendezvous), gloo rejecting the stacked `all_gather_single` output form (added
  `all_gather_stacked`), and a benchmark output checksum that could have contaminated memory
  measurements (moved after measurement, computed in chunks).

## Details of the 2026-11-16 UTC GPU session

The assistant reused the node's preinstalled NVIDIA PyTorch/CUDA build through an isolated venv and
used exactly two GB200 GPUs. It fixed collective API compatibility, PCI-domain and styled-topology
parsing, hardware-gate failure exit status, an overly strict injected-OOM test assumption, stale TSP
attention views, and duplicate/misclassified profiler events. Correctness tolerances were unchanged.
CPU tests, two-GPU FP32/BF16 correctness, one-GPU smoke checks, and before/after memory measurements
were executed on the node. Source patches, hashes, environment records, and measurements are saved
under `results/`; the shared GPU runtime was not replaced.

The assistant completed all 132 matched and ablation launches, generated tables and plots, and saved
the raw measurements, analysis, and reproduction documentation locally. These are block-level
forward measurements, not training or decoding results.
The Markdown report is now complete. A final PDF/LaTeX report, memory-limit sweep, human review,
and earlier-session transcript exports remain outstanding. The legacy `report/report.tex` remains
a skeleton, not the current measured report.

For the subsequent backend investigation, the assistant verified that the installed lower-right
`CausalBias` path bypasses cuDNN, then implemented native bottom-right cuDNN SDPA with bounded
metadata-only caching and correct stream/lifetime handling. The replacement worker passed 120
regression tests, 289 standard GPU correctness records, 133 native-cuDNN records, and 54 benchmark
launches across SDPA, Flash, and native cuDNN. The original results were preserved, all new controls
used the same replacement worker, and completed sweeps were checkpointed locally. See
`results/attention_dispatch/README.md`. No full K/V/weight overlap pipeline or backward implementation
was added.

The combined native-cuDNN bucketing follow-up passed 97 dedicated correctness records and all 63
benchmark launches. It confirmed stable allocation savings for attention and MLP, with substantial
latency costs for small attention buckets. Full data and limitations are in
`results/native_bucketing/README.md`. The report summarizes the current evidence without conflating
datasets from different workers or claiming the same speedup for attention and MLP.

## Logs and redaction

- Current transcript: actual `user.message` and visible `assistant.message` events, plus visible
  `task_complete` summaries. Empty administrative user events, system instructions, private reasoning,
  internal state, and raw tool payloads are excluded. The transcript header records its cutoff and
  counts; the tool inventory is aggregate, not a full command transcript.
- Explicit redactions cover GPU job identifiers, personal home-directory usernames, and temporary
  workspace identifiers. Visible wording is otherwise retained, with line endings/trailing whitespace
  normalized. No unavailable dialogue was invented.
- The report delivery and commit/push request are tracked above; the packaging completion acknowledgement
  and future messages fall after the saved transcript snapshot. Refresh the export when more work is
  performed; do not treat a snapshot as perpetual coverage.
- Earlier paper/specification and implementation transcripts must still be supplied from their
  original sessions. The current machine's repository-session history exposes only this CLI session.
- The report was print-checked as two pages on A4 and US Letter at 11 pt, 1.15 line spacing, and
  0.65-inch margins. Markdown pagination depends on renderer settings. Rendering tools were installed
  only in a private temporary environment; no project or shared-runtime dependencies changed.
- Before publication to the public repository, internal GPU hostnames and private network addresses
  were redacted from the three saved NCCL hardware-check logs. Each log declares these redactions;
  timings, bandwidth measurements, GPU indices/PCI identities, and recorded source hashes are unchanged.
