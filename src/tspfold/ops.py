"""Kernels shared by TP and TSP: bottom-right causal attention and one SwiGLU weight shard.

Tensor layout for attention is [B, L, H, d] (Algorithm 1 line 11).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from .config import ConfigError, check_attn_backend

_SDPA_RESTRICTED = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def sdpa_backend_context(backend: str) -> contextlib.AbstractContextManager:
    """Restrict SDPA to one backend for the duration of a forward call ("sdpa" = PyTorch's choice).

    Applied once per block forward so per-call context overhead is identical across methods.
    """
    check_attn_backend(backend)
    if backend in _SDPA_RESTRICTED:
        return sdpa_kernel([_SDPA_RESTRICTED[backend]])
    return contextlib.nullcontext()


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str = "sdpa") -> torch.Tensor:
    """Bottom-right-aligned causal attention.

    q: [B, Lq, H, d]; k, v: [B, Lk, H, d] with Lk >= Lq. Query i sits at position (Lk - Lq) + i of the
    key sequence and attends keys j <= i + (Lk - Lq). For Lq == Lk this is ordinary causal attention.
    Returns [B, Lq, H, d] (possibly a strided view).

    Note: F.scaled_dot_product_attention(is_causal=True) is TOP-left aligned when Lq != Lk, so it is
    only used for square inputs; non-square inputs use torch's lower-right CausalBias, which dispatches
    to the flash kernel (causal == lower-right there) or the mem-efficient kernel's lower-right mask.
    """
    lq, lk = q.size(1), k.size(1)
    if lk < lq:
        raise ConfigError(f"key length {lk} shorter than query length {lq}")
    if backend == "naive":
        return naive_causal_attention(q, k, v)
    if backend == "flash_attn":
        from flash_attn import flash_attn_func  # optional dependency; causal is bottom-right since v2.1

        return flash_attn_func(q, k, v, causal=True)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if lq == lk:
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    else:
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=causal_lower_right(lq, lk))
    return out.transpose(1, 2)


def naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Explicit masked-softmax attention (same alignment contract as causal_attention)."""
    lq, lk, d = q.size(1), k.size(1), q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (1.0 / math.sqrt(d))
    qpos = torch.arange(lq, device=q.device)[:, None] + (lk - lq)
    kpos = torch.arange(lk, device=q.device)[None, :]
    scores = scores.masked_fill(kpos > qpos, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def swiglu_shard(x2d: torch.Tensor, packed: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """One intermediate-channel shard of the gated MLP, accumulated into `out`.

    x2d: [N, h]; packed: [W_gate^(r); W_up^(r); W_down^(r)^T] of shape [3*I_s, h].
    Returns out + (SiLU(x W_gate^T) * (x W_up^T)) W_down^(r)^T  (Algorithm 2 lines 12-14).
    Gate and up share one GEMM; SiLU and the product run in place on the GEMM output.
    """
    i_s = packed.size(0) // 3
    gate_up = F.linear(x2d, packed[: 2 * i_s])
    gated = F.silu(gate_up[:, :i_s], inplace=True)
    gated.mul_(gate_up[:, i_s:])
    w_down_t = packed[2 * i_s :]
    if out is None:
        return torch.mm(gated, w_down_t)
    return out.addmm_(gated, w_down_t)
