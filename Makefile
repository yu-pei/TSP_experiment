SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

PY ?= .venv/bin/python
NPROC ?= 2
RESULTS ?= results
SCRATCH ?= scratch
# CPU threads for the float64 reference in the GPU correctness run (torchrun defaults to 1)
REF_THREADS ?= 8
PORT ?= $(shell $(PY) -c "from tspfold.launch import free_port; print(free_port())")
TORCHRUN = $(PY) -m torch.distributed.run --nnodes=1 --nproc_per_node=$(NPROC) --master_addr=127.0.0.1 --master_port=$(PORT)

.PHONY: help test hw-check correctness-gpu gpu-smoke sweep-main sweep-long sweep-ablation analyze report

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

test: ## Unit tests + gloo (CPU) distributed correctness, D=1 and D=2
	$(PY) -m pytest -q

hw-check: ## T02: environment manifest, topology, NVLink gate, collective/P2P sanity, bandwidth
	mkdir -p $(RESULTS)/env
	NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH $(TORCHRUN) -m tspfold.hwcheck --out-dir $(RESULTS)/env 2>&1 | tee $(RESULTS)/env/hwcheck.log

correctness-gpu: ## T08 on GPUs: D=2 suite (FP32 strict + BF16) and D=1 smoke
	mkdir -p $(RESULTS)/correctness
	OMP_NUM_THREADS=$(REF_THREADS) $(TORCHRUN) -m tspfold.correctness --suite gpu --out $(RESULTS)/correctness/gpu_d2.jsonl 2>&1 | tee $(RESULTS)/correctness/gpu_d2.log
	$(PY) -m tspfold.correctness --suite smoke --out $(RESULTS)/correctness/smoke_d1.jsonl 2>&1 | tee $(RESULTS)/correctness/smoke_d1.log

gpu-smoke: ## Quick GPU pipeline check at benchmark size (scratch output, not a result)
	$(PY) -m tspfold.sweep --config configs/sweep_gpu_smoke.json --results-dir $(SCRATCH)
	$(PY) -m tspfold.analyze --results-dir $(SCRATCH) --out-dir $(SCRATCH)/analysis

sweep-main: ## T11: matched TP vs TSP sweep (B=1 sequence sweep + batch sweep)
	$(PY) -m tspfold.sweep --config configs/sweep_main.json --results-dir $(RESULTS) 2>&1 | tee -a $(RESULTS)/sweep_main.log

sweep-long: ## T11: extend B=1 toward the memory limit
	$(PY) -m tspfold.sweep --config configs/sweep_long.json --results-dir $(RESULTS) 2>&1 | tee -a $(RESULTS)/sweep_long.log

sweep-ablation: ## T10/T11: TSP overlap on/off and head-bucket ablation
	$(PY) -m tspfold.sweep --config configs/sweep_ablation.json --results-dir $(RESULTS) 2>&1 | tee -a $(RESULTS)/sweep_ablation.log

analyze: ## T12: summary tables, matched ratios, figures
	$(PY) -m tspfold.analyze --results-dir $(RESULTS) --out-dir $(RESULTS)/analysis

report: ## T13: build report/report.pdf (needs a LaTeX engine) and check the 2-page limit
	cd report && latexmk -pdf -interaction=nonstopmode report.tex
	@pages=$$(pdfinfo report/report.pdf | awk '/^Pages:/{print $$2}'); echo "pages: $$pages"; test $$pages -le 2
