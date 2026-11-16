"""Distributed correctness on CPU/gloo through the same CLI used on GPUs (torchrun ... tspfold.correctness)."""

import json
import os
import subprocess
import sys

import pytest

from tspfold.correctness import build_suite
from tspfold.launch import free_port, torchrun_command

pytestmark = pytest.mark.distributed

TIMEOUT_S = 900


def _run(cmd: list[str], out, env=None) -> list[dict]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
    if not out.exists():
        pytest.fail(f"no results written (exit {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-4000:]}")
    return [json.loads(line) for line in out.read_text().splitlines()]


@pytest.fixture(scope="module")
def d2_records(tmp_path_factory):
    out = tmp_path_factory.mktemp("correctness") / "cpu_d2.jsonl"
    cmd = torchrun_command("tspfold.correctness", ["--suite", "cpu", "--device", "cpu", "--out", str(out)])
    return _run(cmd, out)


@pytest.fixture(scope="module")
def d1_records(tmp_path_factory):
    out = tmp_path_factory.mktemp("correctness") / "smoke_d1.jsonl"
    env = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())}
    cmd = [sys.executable, "-m", "tspfold.correctness", "--suite", "smoke", "--device", "cpu", "--out", str(out)]
    return _run(cmd, out, env=env)


CPU_CASES = build_suite("cpu", 2)


def test_case_ids_unique():
    ids = [c.case_id for c in CPU_CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CPU_CASES, ids=lambda c: c.case_id)
def test_d2_case(d2_records, case):
    recs = {r["method"]: r for r in d2_records if r["case_id"] == case.case_id}
    assert set(recs) == {"tp", "tsp", "unsharded"}
    for method in ("tp", "tsp"):
        r = recs[method]
        assert r["passed"], f"{method}: {r['problems']} max_abs={r['max_abs_err']:.3e} l2={r['l2_rel_err']:.3e}"
        if case.perturb:
            assert len(r["perturb"]) >= 2


def test_d2_ring_exchange(d2_records):
    (ring,) = [r for r in d2_records if r["case_id"] == "ring_exchange"]
    assert ring["passed"]


def test_d1_smoke(d1_records):
    methods = [r for r in d1_records if r["method"] in ("tp", "tsp")]
    assert len(methods) == 2 * len(build_suite("smoke", 1))
    for r in methods:
        assert r["world_size"] == 1
        assert r["passed"], f"{r['case_id']} {r['method']}: {r['problems']}"
