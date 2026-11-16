# AI-use record

Running declaration required by the specification (sections 2 and 8). Update it whenever an AI tool
contributes to the project, and store the matching exported chat logs in [`ai_logs/`](../ai_logs/).
Do not describe the implementation as independently authored.

## Summary

| Phase | AI role | Human role | Chat log |
| --- | --- | --- | --- |
| Paper discussion and project specification (`TSP_PROJECT_SPEC.md`) | AI-assisted, as stated in the specification itself (literature interpretation, specification drafting) | Directed the discussion; adopted the specification | Earlier sessions; **not yet exported to `ai_logs/`** |
| Repository creation and local implementation, 2026-11-15 | AI coding assistant (Copilot SDK in VS Code) wrote essentially all code, tests, configs, scripts, documentation, and the report skeleton, and ran the local validation described below | Requested the work ("start the repo and implementation locally ... complete as much as possible before moving to GPUs"); review of the AI-written code is **pending** | This session; **must be exported** to `ai_logs/` |
| GPU execution, analysis, report writing | Not yet performed | | |

## Details of the 2026-11-15 implementation session

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

What has not been done by anyone yet: GPU/NCCL execution, NVLink verification, measurements,
analysis of real data, and the report text for results and observations.

## Logs and redaction

- Export each AI session transcript (user and assistant messages; tool actions where the export
  includes them) into `ai_logs/` using the naming scheme in [`ai_logs/README.md`](../ai_logs/README.md).
- Logs that are unavailable must be listed as unavailable here, never reconstructed.
- Before committing, review logs for credentials, tokens, hostnames, usernames in paths, and other
  private information; mark redactions inline as `[REDACTED: reason]`.
