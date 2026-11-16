# Results

Produced on the GPU host by the Makefile targets; commit everything here except `*/status/`.

| Path | Producer | Content |
| --- | --- | --- |
| `env/` | `make hw-check` | `env_manifest.json`, `nvidia_smi_topo.txt`, `nvidia_smi_nvlink.txt`, `hw_gate.json`, `hwcheck.log` (NCCL_DEBUG=INFO) |
| `correctness/` | `make correctness-gpu` | `gpu_d2.jsonl`, `smoke_d1.jsonl` (+ logs): per case and method error metrics and checks |
| `<sweep>/raw/` | `make sweep-*` | one JSONL per launched case (`schema: tspfold.bench/1`), including failure outcomes |
| `<sweep>/logs/` | `make sweep-*` | full stdout/stderr per case; `<sweep>/commands.txt` lists exact launch commands |
| `analysis/` | `make analyze` | `summary.md/csv`, `matched.csv`, `outcomes.csv`, figures |

CPU/gloo runs (for example `configs/sweep_cpu_smoke.json`) only validate the harness; write them to a
scratch directory, never here.
