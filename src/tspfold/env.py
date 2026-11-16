"""Hardware/software provenance recorded with every result (spec section 6, hardware gate)."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

import torch

from .dist_utils import precision_settings

_ENV_PREFIXES = ("NCCL_", "TORCH_NCCL_", "CUDA_", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")


def run_cmd(cmd: list[str], timeout: float = 30.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_info(repo_dir: str | None = None) -> dict:
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rev = run_cmd(["git", "-C", repo_dir, "rev-parse", "HEAD"])
    status = run_cmd(["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"])
    return {"rev": rev.strip() if rev else None, "dirty": bool(status.strip()) if status is not None else None}


def nvidia_smi_gpus() -> list[dict]:
    out = run_cmd(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"]
    )
    gpus = []
    for line in (out or "").strip().splitlines():
        idx, name, driver, mem, bus = (s.strip() for s in line.split(","))
        gpus.append({"index": int(idx), "name": name, "driver": driver, "memory_total": mem, "pci_bus_id": bus})
    return gpus


def _norm_bus_id(bus: str) -> str:
    # Normalize domain padding without merging devices in distinct PCI domains (e.g. GB200).
    if bus.count(":") == 2:
        domain, address = bus.lower().split(":", 1)
        return f"{int(domain, 16):08x}:{address}"
    return bus.lower()


def visible_to_physical() -> list[int | None]:
    """Physical nvidia-smi index of each visible CUDA device (matched by PCI bus id when possible)."""
    if not torch.cuda.is_available():
        return []
    smi = {_norm_bus_id(g["pci_bus_id"]): g["index"] for g in nvidia_smi_gpus()}
    result = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        bus = None
        if hasattr(props, "pci_bus_id"):
            bus = f"{getattr(props, 'pci_domain_id', 0):04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        result.append(smi.get(_norm_bus_id(bus)) if bus else None)
    return result


def parse_topology(topo: str) -> dict[tuple[int, int], str]:
    """Parse `nvidia-smi topo -m` into {(gpu_i, gpu_j): link} for GPU-GPU entries (e.g. 'NV12', 'PIX', 'SYS')."""
    topo = re.sub(r"\x1b\[[0-9;]*m", "", topo)  # nvidia-smi underlines headers when TERM is set.
    lines = [ln.rstrip() for ln in topo.splitlines() if ln.strip()]
    header = next((ln for ln in lines if re.search(r"\bGPU0\b", ln)), None)
    if header is None:
        return {}
    cols = [c for c in header.split() if re.fullmatch(r"GPU\d+", c)]
    links = {}
    for ln in lines:
        parts = ln.split()
        if not parts or not re.fullmatch(r"GPU\d+", parts[0]) or ln is header:
            continue
        i = int(parts[0][3:])
        for col, val in zip(cols, parts[1 : 1 + len(cols)]):
            links[(i, int(col[3:]))] = val
    return links


def collect_env(device: torch.device | None = None) -> dict:
    cuda = torch.cuda.is_available()
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": _version("numpy"),
        "flash_attn": _version("flash-attn") or _version("flash_attn"),
        "cudnn_frontend": _version("nvidia-cudnn-frontend"),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
        "git": git_info(),
        "precision": precision_settings(),
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)},
    }
    if cuda:
        info["sdpa_enabled"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        }
        info["visible_gpus"] = [
            {
                "visible_index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "sm_count": torch.cuda.get_device_properties(i).multi_processor_count,
            }
            for i in range(torch.cuda.device_count())
        ]
        info["visible_to_physical"] = visible_to_physical()
        smi = nvidia_smi_gpus()
        info["driver"] = smi[0]["driver"] if smi else None
        info["nvidia_smi_gpus"] = smi
    return info
