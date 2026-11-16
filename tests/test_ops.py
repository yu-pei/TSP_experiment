import gc
import math
import weakref

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


def test_cudnn_frontend_rejects_cpu_inputs():
    q, k, v = (torch.randn(1, 4, 2, 8) for _ in range(3))
    with pytest.raises(RuntimeError, match="requires CUDA FP16/BF16"):
        causal_attention(q, k, v, "cudnn_frontend")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,lq,lk", [(1, 16, 16), (2, 16, 64), (1, 3, 7)])
def test_cudnn_frontend_strided_lower_right(dtype, batch, lq, lk):
    pytest.importorskip("cudnn")
    torch.manual_seed(123)
    qkv = torch.randn(batch, 2 * lq, 3, 8, 64, dtype=dtype, device="cuda")
    kv = torch.randn(2, batch, lk + lq, 8, 64, dtype=dtype, device="cuda")
    q, k, v = qkv[:, lq:, 0, 4:], kv[0, :, :lk, 4:], kv[1, :, :lk, 4:]
    out = causal_attention(q, k, v, "cudnn_frontend")
    ref = _explicit(q.double().cpu(), k.double().cpu(), v.double().cpu(), lk - lq)
    torch.testing.assert_close(out.double().cpu(), ref, atol=0.02, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cudnn_frontend_cache_stream_and_input_lifetimes():
    pytest.importorskip("cudnn")
    baseline = torch.cuda.memory_allocated()
    q, k, v = (torch.randn(1, length, 2, 64, device="cuda", dtype=torch.bfloat16)
               for length in (16, 64, 64))
    first = causal_attention(q, k, v, "cudnn_frontend")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # A cached plan must execute on this stream, after the new input has been produced.
        torch.cuda._sleep(5_000_000)
        changed = v + 1
        second = causal_attention(q, k, changed, "cudnn_frontend")
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(second, first + 1, atol=0.02, rtol=0.02)
    refs = [weakref.ref(t) for t in (q, k, v, changed)]
    del q, k, v, changed, first, second
    torch.cuda.synchronize()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert torch.cuda.memory_allocated() == baseline


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cudnn_frontend_does_not_silently_drop_gradients():
    pytest.importorskip("cudnn")
    q, k, v = (torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
               for _ in range(3))
    with pytest.raises(RuntimeError, match="forward-only"):
        causal_attention(q, k, v, "cudnn_frontend")
