"""Unsharded single-device references used as the correctness oracle (not a performance baseline).

Computed by default in float64 from the exact (already dtype-rounded) global inputs and weights,
with an explicit global-position causal mask, independent of the SDPA/flash code paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .weights import AttnWeights, MLPWeights


def reference_attention(
    x: torch.Tensor,
    w: AttnWeights,
    n_heads: int,
    compute_dtype: torch.dtype = torch.float64,
    heads_per_step: int = 4,
) -> torch.Tensor:
    """Causal MHA y = Attn(x W_Q^T, x W_K^T, x W_V^T) W_O^T on x: [B, S, h]."""
    x = x.to(compute_dtype)
    w = w.to(device=x.device, dtype=compute_dtype)
    bsz, seq, hidden = x.shape
    d = hidden // n_heads
    q = (x @ w.wq.t()).view(bsz, seq, n_heads, d).transpose(1, 2)
    k = (x @ w.wk.t()).view(bsz, seq, n_heads, d).transpose(1, 2)
    v = (x @ w.wv.t()).view(bsz, seq, n_heads, d).transpose(1, 2)
    pos = torch.arange(seq, device=x.device)
    future = pos[None, :] > pos[:, None]  # key position > query position
    out = torch.empty_like(q)
    for h0 in range(0, n_heads, heads_per_step):  # bound the [B, heads, S, S] score buffer
        hs = slice(h0, h0 + heads_per_step)
        scores = (q[:, hs] @ k[:, hs].transpose(-1, -2)) / math.sqrt(d)
        scores = scores.masked_fill(future, float("-inf"))
        out[:, hs] = torch.softmax(scores, dim=-1) @ v[:, hs]
    return out.transpose(1, 2).reshape(bsz, seq, hidden) @ w.wo.t()


def reference_mlp(x: torch.Tensor, w: MLPWeights, compute_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """SwiGLU y = (SiLU(x W_gate^T) * (x W_up^T)) W_down^T on x: [B, S, h]."""
    x = x.to(compute_dtype)
    w = w.to(device=x.device, dtype=compute_dtype)
    return (F.silu(x @ w.w_gate.t()) * (x @ w.w_up.t())) @ w.w_down.t()
