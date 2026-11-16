import math

import pytest
import torch
import torch.nn.functional as F

from tspfold.ops import causal_attention, naive_causal_attention, sdpa_backend_context
from tspfold.reference import reference_attention
from tspfold.weights import make_attention_weights, make_input


def _explicit(q, k, v, q_start):
    """Attention where query i has global position q_start + i and keys sit at 0..Lk-1."""
    d = q.size(-1)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(d)
    qpos = q_start + torch.arange(q.size(1))[:, None]
    s = s.masked_fill(torch.arange(k.size(1))[None, :] > qpos, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)


@pytest.mark.parametrize("backend", ["sdpa", "math", "naive"])
@pytest.mark.parametrize("lq,lk", [(4, 4), (4, 8), (3, 12), (1, 5)])
def test_bottom_right_alignment(backend, lq, lk):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, lq, 3, 8), torch.randn(2, lk, 3, 8), torch.randn(2, lk, 3, 8)
    with sdpa_backend_context(backend if backend != "naive" else "sdpa"):
        out = causal_attention(q, k, v, backend)
    torch.testing.assert_close(out, _explicit(q, k, v, lk - lq), rtol=1e-5, atol=1e-6)


def test_generic_causal_flag_is_top_left_for_non_square():
    # Regression guard for the pitfall the spec warns about: SDPA's is_causal uses top-left alignment
    # when Lq != Lk, which would let TSP's later query chunk miss most of its causal prefix.
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 2, 8), torch.randn(1, 8, 2, 8), torch.randn(1, 8, 2, 8)
    top_left = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    torch.testing.assert_close(top_left, _explicit(q, k, v, 0), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(top_left, causal_attention(q, k, v), atol=1e-3)


def test_naive_matches_sdpa_square():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 6, 2, 4, dtype=torch.float64) for _ in range(3))
    torch.testing.assert_close(naive_causal_attention(q, k, v), causal_attention(q, k, v, "sdpa"))


def test_reference_attention_is_causal_and_matches_mha():
    h, H, S = 16, 4, 12
    w = make_attention_weights(h, seed=0, dtype=torch.float64)
    x = make_input(2, S, h, seed=0, dtype=torch.float64)
    y = reference_attention(x, w, H)
    # Independent formulation via torch's MultiheadAttention-style math per head.
    q, k, v = (x @ m.t() for m in (w.wq, w.wk, w.wv))
    heads = []
    for i in range(H):
        sl = slice(i * h // H, (i + 1) * h // H)
        heads.append(F.scaled_dot_product_attention(q[..., sl], k[..., sl], v[..., sl], is_causal=True))
    torch.testing.assert_close(y, torch.cat(heads, -1) @ w.wo.t())
    t = 5
    x2 = x.clone()
    x2[:, t:] += 1.0
    y2 = reference_attention(x2, w, H)
    assert torch.equal(y2[:, :t], y[:, :t])
    assert not torch.allclose(y2[:, t:], y[:, t:])
