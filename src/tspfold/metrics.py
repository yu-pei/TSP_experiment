"""Error metrics for comparing a method's output against the reference (spec section 5)."""

from __future__ import annotations

import torch

# Initial acceptance thresholds (atol, rtol) from the spec; fp64 is only used for self-checks.
TOLERANCES: dict[str, tuple[float, float]] = {
    "fp64": (1e-10, 1e-8),
    "fp32": (1e-5, 1e-4),
    "bf16": (2e-2, 2e-2),
    "fp16": (2e-2, 2e-2),
}


def compare(y: torch.Tensor, y_ref: torch.Tensor, atol: float, rtol: float, eps: float = 1e-12) -> dict:
    """Max abs error, normalized L2 error norm(y - y_ref) / max(norm(y_ref), eps), and allclose."""
    if y.shape != y_ref.shape:
        raise ValueError(f"shape mismatch {tuple(y.shape)} vs {tuple(y_ref.shape)}")
    y64 = y.detach().to("cpu", torch.float64)
    r64 = y_ref.detach().to("cpu", torch.float64)
    diff = (y64 - r64).abs()
    ref_norm = r64.norm().item()
    return {
        "max_abs_err": diff.max().item(),
        "l2_rel_err": (y64 - r64).norm().item() / max(ref_norm, eps),
        "allclose": bool(torch.allclose(y64, r64, atol=atol, rtol=rtol)),
        # >1 means some element violates |y - ref| <= atol + rtol*|ref|
        "worst_tol_ratio": (diff / (atol + rtol * r64.abs())).max().item(),
        "finite": bool(torch.isfinite(y64).all().item()),
        "ref_norm": ref_norm,
        "ref_max_abs": r64.abs().max().item(),
        "atol": atol,
        "rtol": rtol,
    }


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().to("cpu", torch.float64) - b.detach().to("cpu", torch.float64)).abs().max().item()
