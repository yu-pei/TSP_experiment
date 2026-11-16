"""Zigzag sequence layout (Algorithm 1 line 1; spec section 3).

The global sequence is cut into 2D equal contiguous chunks of length S_c = S/(2D).
Rank p owns chunks p and 2D-1-p, stored back-to-back: local = [chunk_p ; chunk_{2D-1-p}].
"""

from __future__ import annotations

import torch

from .config import ConfigError


def zigzag_chunk_ids(rank: int, world_size: int) -> tuple[int, int]:
    if not 0 <= rank < world_size:
        raise ConfigError(f"rank {rank} out of range for world_size {world_size}")
    return rank, 2 * world_size - 1 - rank


def _check_divisible(seq_len: int, world_size: int) -> int:
    if world_size < 1 or seq_len % (2 * world_size):
        raise ConfigError(f"seq_len={seq_len} not divisible by 2D={2 * world_size}")
    return seq_len // (2 * world_size)


def zigzag_positions(rank: int, world_size: int, seq_len: int) -> torch.Tensor:
    """Global token positions of rank's local tokens, in local order."""
    sc = _check_divisible(seq_len, world_size)
    a, b = zigzag_chunk_ids(rank, world_size)
    return torch.cat([torch.arange(a * sc, (a + 1) * sc), torch.arange(b * sc, (b + 1) * sc)])


def zigzag_shard(x: torch.Tensor, rank: int, world_size: int, dim: int = 1) -> torch.Tensor:
    """Global [.., S, ..] -> rank's local [.., S/D, ..] (contiguous copy)."""
    sc = _check_divisible(x.size(dim), world_size)
    a, b = zigzag_chunk_ids(rank, world_size)
    return torch.cat([x.narrow(dim, a * sc, sc), x.narrow(dim, b * sc, sc)], dim=dim)


def zigzag_unshard(shards: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
    """Inverse of zigzag_shard: rank-ordered local shards -> global sequence order."""
    world_size = len(shards)
    if world_size == 0:
        raise ConfigError("no shards given")
    local = shards[0].size(dim)
    if local % 2 or any(s.size(dim) != local for s in shards):
        raise ConfigError("shards must have equal, even local sequence lengths")
    sc = local // 2
    chunks: list[torch.Tensor | None] = [None] * (2 * world_size)
    for rank, s in enumerate(shards):
        a, b = zigzag_chunk_ids(rank, world_size)
        chunks[a] = s.narrow(dim, 0, sc)
        chunks[b] = s.narrow(dim, sc, sc)
    return torch.cat(chunks, dim=dim)


def zigzag_reorder_gathered(gathered: torch.Tensor, seq_dim: int) -> torch.Tensor:
    """ZigZagReorder (Algorithm 1 line 17).

    `gathered` is an all-gather result stacked along a new leading rank dim: [D, *local_shape],
    where local_shape has the local sequence at `seq_dim` (an index into local_shape).
    Returns the global-order tensor of shape local_shape with the sequence dim scaled by D,
    using a single concatenation kernel.
    """
    world_size = gathered.size(0)
    local = gathered.size(seq_dim + 1)
    if local % 2:
        raise ConfigError(f"local sequence length {local} must be even")
    sc = local // 2
    first = [gathered[r].narrow(seq_dim, 0, sc) for r in range(world_size)]
    second = [gathered[r].narrow(seq_dim, sc, sc) for r in reversed(range(world_size))]
    return torch.cat(first + second, dim=seq_dim)


def causal_key_bounds(rank: int, world_size: int, seq_len: int) -> tuple[int, int]:
    """(s1, s2) of Algorithm 1 line 12: key-prefix lengths for the two local query chunks."""
    sc = _check_divisible(seq_len, world_size)
    return (rank + 1) * sc, (2 * world_size - rank) * sc
