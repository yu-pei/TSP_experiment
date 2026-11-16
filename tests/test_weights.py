import pytest
import torch

from tspfold.config import ConfigError
from tspfold.layout import zigzag_chunk_ids, zigzag_shard
from tspfold.ops import swiglu_shard
from tspfold.reference import reference_mlp
from tspfold.weights import (
    make_attention_weights,
    make_input,
    make_input_range,
    make_mlp_weights,
    reconstruct_attention,
    reconstruct_mlp,
    shard_attention,
    shard_mlp,
    unpack_attention,
    unpack_mlp,
)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_attention_shards_round_trip(world):
    h, H = 32, 8
    w = make_attention_weights(h, seed=3)
    shards = [shard_attention(w, r, world, H) for r in range(world)]
    assert all(s.shape == (4 * h // world, h) and s.is_contiguous() for s in shards)
    back = reconstruct_attention(shards)
    for name in ("wq", "wk", "wv", "wo"):
        assert torch.equal(getattr(back, name), getattr(w, name))


@pytest.mark.parametrize("world", [1, 2, 4])
def test_mlp_shards_round_trip(world):
    h, I = 16, 64
    w = make_mlp_weights(h, I, seed=5)
    shards = [shard_mlp(w, r, world) for r in range(world)]
    assert all(s.shape == (3 * I // world, h) and s.is_contiguous() for s in shards)
    back = reconstruct_mlp(shards)
    for name in ("w_gate", "w_up", "w_down"):
        assert torch.equal(getattr(back, name), getattr(w, name))


def test_output_weight_orientation():
    # Rank p's head-shard attention output A_p [N, h/D] must be projected with W_O[:, cols_p]^T and
    # the D partial projections must sum to the full A @ W_O^T.
    h, H, world = 16, 4, 2
    w = make_attention_weights(h, seed=1)
    a = torch.randn(5, h)
    total = torch.zeros(5, h)
    for r in range(world):
        wo_t = unpack_attention(shard_attention(w, r, world, H))[3]
        cols = slice(r * h // world, (r + 1) * h // world)
        assert torch.equal(wo_t, w.wo[:, cols].t())
        total += a[:, cols] @ wo_t
    torch.testing.assert_close(total, a @ w.wo.t())


def test_swiglu_shards_sum_to_reference():
    h, I, world = 16, 64, 2
    w = make_mlp_weights(h, I, seed=2, dtype=torch.float64)
    x = torch.randn(3, 7, h, dtype=torch.float64)
    y = None
    for r in range(world):
        gate, up, down_t = unpack_mlp(shard_mlp(w, r, world))
        cols = slice(r * I // world, (r + 1) * I // world)
        assert torch.equal(down_t, w.w_down[:, cols].t()) and torch.equal(gate, w.w_gate[cols])
        y = swiglu_shard(x.reshape(-1, h), shard_mlp(w, r, world), y)
    torch.testing.assert_close(y.view(3, 7, h), reference_mlp(x, w), rtol=1e-12, atol=1e-12)


def test_deterministic_generation_and_bad_shards():
    a = make_attention_weights(8, seed=0)
    b = make_attention_weights(8, seed=0)
    assert torch.equal(a.wq, b.wq) and not torch.equal(a.wq, make_attention_weights(8, seed=1).wq)
    with pytest.raises(ConfigError):
        shard_attention(a, 0, 3, n_heads=4)  # 4 heads over 3 ranks
    with pytest.raises(ConfigError):
        unpack_mlp(torch.zeros(4, 8))


@pytest.mark.parametrize("world,seq_len", [(1, 600), (2, 1200), (2, 8)])
def test_rank_local_input_generation_matches_global_tensor(world, seq_len):
    # The benchmark builds each TSP rank's input from its zigzag chunks only; it must equal the
    # zigzag shard of the single global tensor that TP and the reference use.
    full = make_input(2, seq_len, 4, seed=9, dtype=torch.bfloat16)
    sc = seq_len // (2 * world)
    for r in range(world):
        local = torch.cat(
            [make_input_range(2, 4, 9, c * sc, (c + 1) * sc, torch.bfloat16) for c in zigzag_chunk_ids(r, world)], 1
        )
        assert torch.equal(local, zigzag_shard(full, r, world))
    assert torch.equal(make_input(2, seq_len // 2, 4, seed=9), make_input(2, seq_len, 4, seed=9)[:, : seq_len // 2])
