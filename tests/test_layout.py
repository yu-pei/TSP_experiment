import pytest
import torch

from tspfold.config import BlockShape, ConfigError
from tspfold.layout import (
    causal_key_bounds,
    zigzag_chunk_ids,
    zigzag_positions,
    zigzag_reorder_gathered,
    zigzag_shard,
    zigzag_unshard,
)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_zigzag_round_trip_and_positions(world):
    S = 8 * world
    x = torch.arange(2 * S * 3, dtype=torch.float32).view(2, S, 3)
    shards = [zigzag_shard(x, r, world) for r in range(world)]
    assert all(s.shape == (2, S // world, 3) for s in shards)
    assert torch.equal(zigzag_unshard(shards), x)
    for r, s in enumerate(shards):
        assert torch.equal(s, x[:, zigzag_positions(r, world, S)])


def test_zigzag_two_ranks_explicit_chunks():
    # D=2 -> 4 chunks; rank 0 owns chunks (0, 3), rank 1 owns (1, 2).
    x = torch.arange(8).view(1, 8)
    assert zigzag_chunk_ids(0, 2) == (0, 3) and zigzag_chunk_ids(1, 2) == (1, 2)
    assert zigzag_shard(x, 0, 2).tolist() == [[0, 1, 6, 7]]
    assert zigzag_shard(x, 1, 2).tolist() == [[2, 3, 4, 5]]
    assert causal_key_bounds(0, 2, 8) == (2, 8)
    assert causal_key_bounds(1, 2, 8) == (4, 6)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_reorder_gathered_matches_global_order(world):
    # Simulate the attention K/V all-gather layout [D, 2(kv), B, S_loc, Hb, d] with seq at local dim 2.
    S = 4 * world
    kv = torch.randn(2, 3, S, 2, 5)
    gathered = torch.stack([zigzag_shard(kv, r, world, dim=2) for r in range(world)])
    out = zigzag_reorder_gathered(gathered, seq_dim=2)
    assert out.is_contiguous()
    assert torch.equal(out, kv)


def test_causal_bounds_cover_each_query_chunk_end():
    world, S = 2, 16
    sc = S // (2 * world)
    for r in range(world):
        s1, s2 = causal_key_bounds(r, world, S)
        pos = zigzag_positions(r, world, S)
        assert s1 == pos[sc - 1].item() + 1  # last query of chunk 1 sees exactly the prefix [0, s1)
        assert s2 == pos[-1].item() + 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seq_len=10),  # not divisible by 2D=4
        dict(n_heads=3, hidden=48),  # H not divisible by D
        dict(intermediate=63),  # I not divisible by D
        dict(hidden=30),  # h not divisible by H
        dict(batch=0),
        dict(world_size=0),
    ],
)
def test_invalid_shapes_rejected(kwargs):
    base = dict(batch=1, seq_len=8, hidden=32, n_heads=4, intermediate=64, world_size=2)
    with pytest.raises(ConfigError):
        BlockShape(**{**base, **kwargs})


def test_invalid_layout_inputs_rejected():
    with pytest.raises(ConfigError):
        zigzag_shard(torch.zeros(1, 6, 2), 0, 2)
    with pytest.raises(ConfigError):
        zigzag_chunk_ids(2, 2)
    with pytest.raises(ConfigError):
        zigzag_unshard([torch.zeros(1, 4), torch.zeros(1, 2)])


def test_head_bucket_validation():
    shape = BlockShape(1, 8, 64, 8, 64, 2)  # 4 heads per rank
    assert shape.check_head_bucket(None) == 4
    assert shape.check_head_bucket(2) == 2
    with pytest.raises(ConfigError):
        shape.check_head_bucket(3)
    with pytest.raises(ConfigError):
        shape.check_head_bucket(0)
