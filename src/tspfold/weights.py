"""Deterministic global weights/inputs and their head/channel shards.

Global weights use the nn.Linear convention W[out, in] (y = x W^T), bias-free:
  attention: W_Q, W_K, W_V, W_O in R^{h x h}
  gated MLP: W_gate, W_up in R^{I x h};  W_down in R^{h x I}

Rank p's shard (identical for TP and TSP) is stored as ONE packed contiguous buffer so the
TSP schedules can move it with a single broadcast / send (Algorithm 1 line 8, Algorithm 2 line 9):
  attention: [W_Q^(p); W_K^(p); W_V^(p); W_O^(p)^T]  shape [4h/D, h]
             where W_*^(p) = W_*[p*h/D:(p+1)*h/D, :] (heads p*H/D..) and W_O^(p) = W_O[:, same cols]
  MLP:       [W_gate^(p); W_up^(p); W_down^(p)^T]     shape [3I/D, h]
             where W_down^(p) = W_down[:, p*I/D:(p+1)*I/D]  (stored transposed, as W_2 in Algorithm 2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from .config import ConfigError

_INPUT_SEED_OFFSET = 1_000_003
_INPUT_BLOCK = 256  # tokens per independently seeded input block


@dataclass
class AttnWeights:
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor

    def to(self, *args, **kwargs) -> "AttnWeights":
        return AttnWeights(*(getattr(self, f.name).to(*args, **kwargs) for f in fields(self)))


@dataclass
class MLPWeights:
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor

    def to(self, *args, **kwargs) -> "MLPWeights":
        return MLPWeights(*(getattr(self, f.name).to(*args, **kwargs) for f in fields(self)))


def _randn(gen: torch.Generator, *shape: int, std: float) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, dtype=torch.float32) * std


def make_attention_weights(hidden: int, seed: int, dtype: torch.dtype = torch.float32) -> AttnWeights:
    """CPU global attention weights with std 1/sqrt(fan_in) so activations stay O(1)."""
    gen = torch.Generator().manual_seed(seed)
    std = 1.0 / math.sqrt(hidden)
    w = [_randn(gen, hidden, hidden, std=std) for _ in range(4)]
    return AttnWeights(*w).to(dtype)


def make_mlp_weights(hidden: int, intermediate: int, seed: int, dtype: torch.dtype = torch.float32) -> MLPWeights:
    gen = torch.Generator().manual_seed(seed + 1)
    w_gate = _randn(gen, intermediate, hidden, std=1.0 / math.sqrt(hidden))
    w_up = _randn(gen, intermediate, hidden, std=1.0 / math.sqrt(hidden))
    w_down = _randn(gen, hidden, intermediate, std=1.0 / math.sqrt(intermediate))
    return MLPWeights(w_gate, w_up, w_down).to(dtype)


def make_input(batch: int, seq_len: int, hidden: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standardized CPU input X ~ N(0, 1) of shape [B, S, h]."""
    return make_input_range(batch, hidden, seed, 0, seq_len, dtype)


def make_input_range(
    batch: int, hidden: int, seed: int, start: int, stop: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Tokens [start, stop) of the global input, i.e. make_input(...)[:, start:stop], without building the rest.

    Each block of _INPUT_BLOCK positions has its own generator seed, so a position's values depend only on
    (seed, batch, hidden, position): every method and rank sees slices of one well-defined global tensor.
    """
    out = torch.empty(batch, stop - start, hidden, dtype=dtype)
    for blk in range(start // _INPUT_BLOCK, (stop - 1) // _INPUT_BLOCK + 1 if stop > start else 0):
        b0 = blk * _INPUT_BLOCK
        gen = torch.Generator().manual_seed((seed + _INPUT_SEED_OFFSET) * 1_000_033 + blk)
        block = torch.randn(batch, _INPUT_BLOCK, hidden, generator=gen, dtype=torch.float32)
        lo, hi = max(start, b0), min(stop, b0 + _INPUT_BLOCK)
        out[:, lo - start : hi - start] = block[:, lo - b0 : hi - b0]
    return out


def _shard_slice(total: int, rank: int, world_size: int) -> slice:
    if total % world_size:
        raise ConfigError(f"dimension {total} not divisible by D={world_size}")
    if not 0 <= rank < world_size:
        raise ConfigError(f"rank {rank} out of range for world_size {world_size}")
    n = total // world_size
    return slice(rank * n, (rank + 1) * n)


def shard_attention(w: AttnWeights, rank: int, world_size: int, n_heads: int) -> torch.Tensor:
    """Packed head shard [W_Q^(p); W_K^(p); W_V^(p); W_O^(p)^T] of shape [4h/D, h]."""
    hidden = w.wq.size(0)
    if hidden % n_heads or n_heads % world_size:
        raise ConfigError(f"cannot split {n_heads} heads of width {hidden} over D={world_size}")
    sl = _shard_slice(hidden, rank, world_size)  # head-aligned because H % D == 0
    return torch.cat([w.wq[sl], w.wk[sl], w.wv[sl], w.wo[:, sl].t()], dim=0).contiguous()


def shard_mlp(w: MLPWeights, rank: int, world_size: int) -> torch.Tensor:
    """Packed channel shard [W_gate^(p); W_up^(p); W_down^(p)^T] of shape [3I/D, h]."""
    sl = _shard_slice(w.w_gate.size(0), rank, world_size)
    return torch.cat([w.w_gate[sl], w.w_up[sl], w.w_down[:, sl].t()], dim=0).contiguous()


def unpack_attention(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Views (W_Q^(r), W_K^(r), W_V^(r), W_O^(r)^T), each [h/D, h]."""
    if packed.dim() != 2 or packed.size(0) % 4:
        raise ConfigError(f"bad packed attention shard shape {tuple(packed.shape)}")
    return tuple(packed.view(4, packed.size(0) // 4, packed.size(1)).unbind(0))  # type: ignore[return-value]


def unpack_mlp(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Views (W_gate^(r), W_up^(r), W_down^(r)^T), each [I/D, h]."""
    if packed.dim() != 2 or packed.size(0) % 3:
        raise ConfigError(f"bad packed MLP shard shape {tuple(packed.shape)}")
    return tuple(packed.view(3, packed.size(0) // 3, packed.size(1)).unbind(0))  # type: ignore[return-value]


def reconstruct_attention(shards: list[torch.Tensor]) -> AttnWeights:
    parts = [unpack_attention(s) for s in shards]
    return AttnWeights(
        wq=torch.cat([p[0] for p in parts], 0),
        wk=torch.cat([p[1] for p in parts], 0),
        wv=torch.cat([p[2] for p in parts], 0),
        wo=torch.cat([p[3].t() for p in parts], 1),
    )


def reconstruct_mlp(shards: list[torch.Tensor]) -> MLPWeights:
    parts = [unpack_mlp(s) for s in shards]
    return MLPWeights(
        w_gate=torch.cat([p[0] for p in parts], 0),
        w_up=torch.cat([p[1] for p in parts], 0),
        w_down=torch.cat([p[2].t() for p in parts], 1),
    )
