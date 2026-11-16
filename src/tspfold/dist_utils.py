"""Process-group, device, and precision helpers."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def global_rank(group: dist.ProcessGroup | None, group_rank: int) -> int:
    """Map a rank within `group` to the global rank torch.distributed APIs expect."""
    if group is None:
        return group_rank
    return dist.get_global_rank(group, group_rank)


def all_gather_stacked(t: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """All-gather a contiguous tensor into [D, *t.shape] (rank-major).

    Uses the concatenated output form [D * t.size(0), ...] that both NCCL and gloo accept, then views it.
    """
    world = dist.get_world_size(group)
    out = t.new_empty((world * t.size(0),) + tuple(t.shape[1:]))
    dist.all_gather_single(out, t, group=group)
    return out.view((world,) + tuple(t.shape))


def init_from_env(device: str = "auto", timeout_s: float = 300.0) -> tuple[int, int, torch.device]:
    """Initialise the default process group from torchrun env vars.

    device="auto" picks cuda:LOCAL_RANK with NCCL when CUDA is available, else CPU with gloo.
    Returns (rank, world_size, device).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    timeout = datetime.timedelta(seconds=timeout_s)
    if use_cuda:
        dev = torch.device("cuda", local_rank)
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", timeout=timeout, device_id=dev)
    else:
        dev = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timeout)
    return dist.get_rank(), dist.get_world_size(), dev


def set_fp32_strict(strict: bool = True) -> dict:
    """Disable (strict=True) or allow TF32 for FP32 matmul/conv; returns the resulting settings."""
    mode = "ieee" if strict else "tf32"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.fp32_precision = mode
    else:  # older torch
        torch.backends.cuda.matmul.allow_tf32 = not strict
        torch.backends.cudnn.allow_tf32 = not strict
    return precision_settings()


def precision_settings() -> dict:
    out = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        out["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
        out["cudnn_fp32_precision"] = torch.backends.cudnn.fp32_precision
    else:
        out["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        out["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    out["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    return out


def synchronize(device: torch.device) -> None:
    """Wait for all queued work on every stream of `device` (incl. NCCL's internal streams)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
