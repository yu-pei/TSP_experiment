import contextlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tspfold import bench, dist_utils, env as env_utils, hwcheck
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
    styled = TOPO_NVLINK.replace("\tGPU0", "\t\x1b[4mGPU0", 1).replace("GPU NUMA ID\n", "GPU NUMA ID\x1b[0m\n", 1)
    assert parse_topology(styled) == links


def test_visible_to_physical_preserves_pci_domains(monkeypatch):
    domains = [8, 9, 24, 25]
    monkeypatch.setattr(env_utils, "nvidia_smi_gpus", lambda: [
        {"index": i, "pci_bus_id": f"{domain:08x}:01:00.0"} for i, domain in enumerate(domains)
    ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(
        pci_domain_id=domains[1 - i], pci_bus_id=1, pci_device_id=0,
    ))
    assert env_utils.visible_to_physical() == [1, 0]


@pytest.mark.parametrize("legacy", [False, True])
def test_all_gather_stacked_collective_api_compatibility(monkeypatch, legacy):
    def gather(out, value, group=None):
        out.copy_(torch.cat([value, value + 10], dim=0))

    api = "all_gather_into_tensor" if legacy else "all_gather_single"
    other = "all_gather_single" if legacy else "all_gather_into_tensor"
    monkeypatch.setattr(dist_utils.dist, api, gather, raising=False)
    monkeypatch.delattr(dist_utils.dist, other, raising=False)
    monkeypatch.setattr(dist_utils.dist, "get_world_size", lambda group: 2)
    value = torch.arange(6).view(2, 3)
    torch.testing.assert_close(dist_utils.all_gather_stacked(value), torch.stack([value, value + 10]))


def test_hwcheck_returns_failure_when_gate_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(hwcheck, "init_from_env", lambda device: (0, 2, torch.device("cpu")))
    monkeypatch.setattr(hwcheck, "collective_sanity", lambda device: {"all_reduce": True})
    monkeypatch.setattr(hwcheck, "bandwidth", lambda *args: [])
    monkeypatch.setattr(hwcheck, "collect_env", lambda device: {})
    monkeypatch.setattr(hwcheck, "run_cmd", lambda cmd: None)
    monkeypatch.setattr(hwcheck.dist, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(hwcheck.dist, "destroy_process_group", lambda: None)
    assert hwcheck.main(["--device", "cpu", "--out-dir", str(tmp_path)]) == 1
    gate = json.loads((tmp_path / "hw_gate.json").read_text())
    assert not gate["gate_pass"] and "CUDA not used" in gate["reasons"]


def test_profile_counts_kernels_not_cuda_annotations(monkeypatch):
    from torch.autograd import DeviceType

    events = [
        SimpleNamespace(
            key=name, device_type=DeviceType.CUDA, count=1,
            self_device_time_total=us, is_user_annotation=annotation,
        )
        for name, us, annotation in [
            ("nvjet_sm100_tst_128x64_64x10_2x2_2cta_h_bz_NNT", 5.0, False),
            ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 12.0, False),
            ("nccl:all_reduce", 12.0, True),
        ]
    ]
    prof = SimpleNamespace(key_averages=lambda: events)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: contextlib.nullcontext(prof))
    monkeypatch.setattr(bench, "synchronize", lambda device: None)
    result = bench.profile_breakdown(lambda x: x, torch.ones(1), torch.device("cuda"), 1)
    assert result["category_us_per_iter"] == {"gemm": 5.0, "comm": 12.0}
    assert len(result["top_kernels"]) == 2


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
    for r in by_status["oom"]:
        assert [s["rank"] for s in r["rank_status"] if s["status"] == "oom"] == [1]
        for s in r["rank_status"]:
            if s["rank"] == 0:  # Gloo may report peer disconnection before torchrun terminates it.
                assert s["status"] == "error" and "Connection closed by peer" in s["error"]

    out = subprocess.run([sys.executable, "-m", "tspfold.analyze", "--results-dir", str(tmp_path),
                          "--out-dir", str(tmp_path / "analysis"), "--no-plots"],
                         capture_output=True, text=True, timeout=120)  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert "4 non-ok outcomes" in out.stdout
