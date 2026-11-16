"""Conventional two-way tensor parallelism baseline (spec section 4).

Inputs are replicated ([B, S, h] on every rank). Rank p holds the same packed shard as TSP:
heads p*H/D.. for attention, intermediate channels p*I/D.. for the gated MLP. Partial outputs of
the row-parallel output/down projection are summed with one all-reduce, so every rank returns the
full replicated output [B, S, h].
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import BlockShape, ConfigError, check_attn_backend
from .ops import causal_attention, sdpa_backend_context, swiglu_shard


def _check(t: torch.Tensor, shape: tuple, what: str) -> None:
    if tuple(t.shape) != tuple(shape):
        raise ConfigError(f"{what}: expected shape {tuple(shape)}, got {tuple(t.shape)}")


class TPAttention:
    def __init__(self, packed: torch.Tensor, shape: BlockShape, *, group=None, backend: str = "sdpa"):
        if dist.get_world_size(group) != shape.world_size:
            raise ConfigError(f"process group size {dist.get_world_size(group)} != D={shape.world_size}")
        _check(packed, (4 * shape.hidden_per_rank, shape.hidden), "packed attention shard")
        self.packed, self.shape, self.group = packed, shape, group
        self.backend = check_attn_backend(backend)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sh = self.shape
        _check(x, (x.size(0), sh.seq_len, sh.hidden), "TP attention input")
        bsz, hs = x.size(0), sh.hidden_per_rank
        with sdpa_backend_context(self.backend):
            qkv = F.linear(x, self.packed[: 3 * hs]).view(bsz, sh.seq_len, 3, sh.heads_per_rank, sh.head_dim)
            attn = causal_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], self.backend)
            y = torch.mm(attn.reshape(bsz * sh.seq_len, hs), self.packed[3 * hs :])
            dist.all_reduce(y, group=self.group)
        return y.view(bsz, sh.seq_len, sh.hidden)


class TPMLP:
    def __init__(self, packed: torch.Tensor, shape: BlockShape, *, group=None):
        if dist.get_world_size(group) != shape.world_size:
            raise ConfigError(f"process group size {dist.get_world_size(group)} != D={shape.world_size}")
        _check(packed, (3 * shape.inter_per_rank, shape.hidden), "packed MLP shard")
        self.packed, self.shape, self.group = packed, shape, group

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sh = self.shape
        _check(x, (x.size(0), sh.seq_len, sh.hidden), "TP MLP input")
        y = swiglu_shard(x.reshape(-1, sh.hidden), self.packed)
        dist.all_reduce(y, group=self.group)
        return y.view(x.size(0), sh.seq_len, sh.hidden)
