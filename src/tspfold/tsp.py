"""Tensor+sequence parallel (TSP) forward schedules: Algorithm 1 (attention) and Algorithm 2 (MLP).

Rank p holds (i) its zigzag token shard [B, S/D, h] (chunks p and 2D-1-p) and (ii) one packed
weight shard (see weights.py). Outputs stay in the same zigzag layout; nothing is gathered here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import BlockShape, ConfigError, check_attn_backend
from .dist_utils import all_gather_stacked, global_rank
from .layout import causal_key_bounds, zigzag_reorder_gathered
from .ops import causal_attention, sdpa_backend_context, swiglu_shard


def _check(t: torch.Tensor, shape: tuple, what: str) -> None:
    if tuple(t.shape) != tuple(shape):
        raise ConfigError(f"{what}: expected shape {tuple(shape)}, got {tuple(t.shape)}")


def _check_group(group, shape: BlockShape) -> tuple[int, int]:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if world != shape.world_size:
        raise ConfigError(f"process group size {world} != D={shape.world_size}")
    return rank, world


def ring_exchange(send: torch.Tensor, recv: torch.Tensor, group=None) -> list:
    """Async ring step: send `send` to rank p+1 and receive into `recv` from rank p-1 (mod D).

    Returns the request handles; `send` must stay alive and unmodified, and `recv` must not be read,
    until every handle has been waited on.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    ops = [
        dist.P2POp(dist.isend, send, global_rank(group, (rank + 1) % world), group),
        dist.P2POp(dist.irecv, recv, global_rank(group, (rank - 1) % world), group),
    ]
    return dist.batch_isend_irecv(ops)


class TSPAttention:
    """Algorithm 1: broadcast packed head shards one owner at a time; all-gather K/V per head bucket.

    overlap=True prefetches the next owner's weight broadcast before computing the current shard
    (the owner itself only reads its persistent buffer, so it does not stall on its own broadcast).
    overlap=False issues each broadcast synchronously, literally following Algorithm 1.
    """

    def __init__(
        self,
        packed: torch.Tensor,
        shape: BlockShape,
        *,
        group=None,
        head_bucket: int | None = None,
        overlap: bool = True,
        backend: str = "sdpa",
    ):
        self.rank, self.world = _check_group(group, shape)
        _check(packed, (4 * shape.hidden_per_rank, shape.hidden), "packed attention shard")
        self.packed, self.shape, self.group = packed, shape, group
        self.head_bucket = shape.check_head_bucket(head_bucket)
        self.overlap = overlap
        self.backend = check_attn_backend(backend)

    def _broadcast(self, owner: int):
        buf = self.packed if owner == self.rank else torch.empty_like(self.packed)
        work = dist.broadcast(buf, src=global_rank(self.group, owner), group=self.group, async_op=True)
        return buf, work

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sh, p, D = self.shape, self.rank, self.world
        _check(x, (x.size(0), sh.local_seq, sh.hidden), "TSP attention input")
        bsz, s_loc, hs = x.size(0), sh.local_seq, sh.hidden_per_rank
        n_local_heads, d, bucket, sc = sh.heads_per_rank, sh.head_dim, self.head_bucket, sh.chunk_len
        s1, s2 = causal_key_bounds(p, D, sh.seq_len)  # line 12
        x2d = x.reshape(-1, sh.hidden)
        y = None
        deferred = []  # owner-side broadcast handles, waited before returning
        with sdpa_backend_context(self.backend):
            prefetched = self._broadcast(0) if self.overlap else None
            for r in range(D):  # line 7: loop over weight-owning ranks
                if self.overlap:
                    w, work = prefetched
                    prefetched = self._broadcast(r + 1) if r + 1 < D else None
                    if r == p:
                        deferred.append(work)
                    else:
                        work.wait()
                else:
                    w, work = self._broadcast(r)  # line 8
                    work.wait()
                # line 9: unpack [W_Q; W_K; W_V] and W_O^T from the packed buffer (views, no copy)
                w_qkv, w_o_t = w[: 3 * hs], w[3 * hs :]
                qkv = F.linear(x2d, w_qkv).view(bsz, s_loc, 3, n_local_heads, d)  # lines 10-11
                attn = torch.empty(bsz, s_loc, n_local_heads, d, dtype=x.dtype, device=x.device)  # line 13
                for b0 in range(0, n_local_heads, bucket):  # line 14: head bucketing
                    heads = slice(b0, b0 + bucket)
                    kv_local = qkv[:, :, 1:3, heads].permute(2, 0, 1, 3, 4).contiguous()  # line 15
                    gathered = all_gather_stacked(kv_local, self.group)  # line 16: [D, 2, B, S_loc, Hb, d]
                    kv = zigzag_reorder_gathered(gathered, seq_dim=2)  # line 17: [2, B, S, Hb, d]
                    del gathered, kv_local
                    q = qkv[:, :, 0, heads]  # line 18: split local queries into the two chunks
                    attn[:, :sc, heads] = causal_attention(q[:, :sc], kv[0, :, :s1], kv[1, :, :s1], self.backend)
                    attn[:, sc:, heads] = causal_attention(q[:, sc:], kv[0, :, :s2], kv[1, :, :s2], self.backend)
                    del kv  # lines 19-21
                del qkv
                attn2d = attn.view(-1, hs)
                y = torch.mm(attn2d, w_o_t) if y is None else y.addmm_(attn2d, w_o_t)  # line 23
                del attn, w, work, w_qkv, w_o_t
        for work in deferred:
            work.wait()
        return y.view(bsz, s_loc, sh.hidden)


class TSPMLP:
    """Algorithm 2: weight shards circulate in a ring; partial outputs accumulate locally.

    overlap=True launches the next ring transfer before the current shard's GEMMs and waits only
    before the received shard is consumed. overlap=False waits for each transfer before computing.
    No activation all-reduce is performed.
    """

    def __init__(self, packed: torch.Tensor, shape: BlockShape, *, group=None, overlap: bool = True):
        self.rank, self.world = _check_group(group, shape)
        _check(packed, (3 * shape.inter_per_rank, shape.hidden), "packed MLP shard")
        self.packed, self.shape, self.group = packed, shape, group
        self.overlap = overlap

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sh, D = self.shape, self.world
        _check(x, (x.size(0), sh.local_seq, sh.hidden), "TSP MLP input")
        x2d = x.reshape(-1, sh.hidden)
        current = self.packed  # line 6
        y = None
        for t in range(D):  # line 7
            pending = []
            if t < D - 1:  # lines 8-10: send to p+1, receive from p-1
                incoming = torch.empty_like(self.packed)
                pending = ring_exchange(current, incoming, self.group)
                if not self.overlap:
                    for req in pending:
                        req.wait()
                    pending = []
            y = swiglu_shard(x2d, current, y)  # lines 12-14
            for req in pending:  # line 16: wait before consuming received data
                req.wait()
            if t < D - 1:
                current = incoming  # line 17; the previous buffer's send has completed
        return y.view(x.size(0), sh.local_seq, sh.hidden)
