import json
import os
import subprocess
import sys

import pytest

from tspfold.env import parse_topology

TOPO_NVLINK = """\
\tGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV18\tSYS\tPXB\t0-55,112-167\t0\t\tN/A
GPU1\tNV18\t X \tNV18\tPXB\t0-55,112-167\t0\t\tN/A
GPU2\tSYS\tNV18\t X \tSYS\t56-111,168-223\t1\t\tN/A
NIC0\tPXB\tPXB\tSYS\t X \t\t\t\t

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NV#  = Connection traversing a bonded set of # NVLinks
"""

TOPO_PCIE = """\
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPHB\t0-31\t0
GPU1\tPHB\t X \t0-31\t0
"""


def test_parse_topology():
    links = parse_topology(TOPO_NVLINK)
    assert links[(0, 1)] == "NV18" and links[(1, 0)] == "NV18"
    assert links[(0, 2)] == "SYS" and links[(1, 2)] == "NV18"
    assert (0, 3) not in links  # NIC columns are not GPUs
    assert parse_topology(TOPO_PCIE)[(0, 1)] == "PHB"
    assert parse_topology("") == {}


@pytest.mark.distributed
def test_sweep_isolates_single_rank_oom(tmp_path):
    cfg = {
        "name": "fault", "hidden": 32, "heads": 4, "intermediate": 64, "dtype": "fp32",
        "warmup": 2, "iters": 2, "trials": 1, "methods": ["tp", "tsp"], "blocks": ["mlp"],
        "grid": [{"batch": [1], "seq_len": [16, 32]}], "timeout_s": 120,
    }  # fmt: skip
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    env = {**os.environ, "TSPFOLD_INJECT_FAULT": "oom@rank1"}
    cmd = [sys.executable, "-m", "tspfold.sweep", "--config", str(tmp_path / "cfg.json"),
           "--results-dir", str(tmp_path), "--device", "cpu"]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = [json.loads(p.read_text()) for p in (tmp_path / "fault" / "raw").glob("*.jsonl")]
    by_status = {}
    for r in recs:
        by_status.setdefault(r["status"], []).append(r)
    # S=16 launches fail on rank 1 only; the blocked peer is torn down and S=32 is not launched.
    assert {r["seq_len"] for r in by_status["oom"]} == {16} and len(by_status["oom"]) == 2
    assert {r["seq_len"] for r in by_status["skipped"]} == {32} and len(by_status["skipped"]) == 2
    assert all(r["elapsed_s"] < 60 for r in by_status["oom"])
    assert all([s["rank"] for s in r["rank_status"]] == [1] for r in by_status["oom"])

    out = subprocess.run([sys.executable, "-m", "tspfold.analyze", "--results-dir", str(tmp_path),
                          "--out-dir", str(tmp_path / "analysis"), "--no-plots"],
                         capture_output=True, text=True, timeout=120)  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert "4 non-ok outcomes" in out.stdout
