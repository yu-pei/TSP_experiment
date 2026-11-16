"""Theoretical per-rank forward communication volume (bytes received per rank), for analysis only.

Accounting (D ranks, beta bytes/element, bandwidth-optimal ring collectives):
  TP  attn/MLP : one all-reduce of [B, S, h]          -> 2 (D-1)/D * B S h beta
  TSP attn     : D-1 packed [4h/D, h] broadcasts       -> (D-1) * 4 h^2 / D * beta
                 D iterations of a K/V all-gather       -> D * (D-1) * 2 B (S/D) (h/D) beta = 2 (D-1) B S h / D * beta
  TSP MLP      : D-1 ring steps of a [3I/D, h] shard   -> (D-1) * 3 I h / D * beta
These are estimates, not measurements; they ignore protocol overheads and latency terms.
"""

from __future__ import annotations

from .config import BlockShape


def comm_bytes(method: str, block: str, shape: BlockShape, bytes_per_elem: int) -> dict[str, float]:
    B, S, h, I, D = shape.batch, shape.seq_len, shape.hidden, shape.intermediate, shape.world_size
    beta = bytes_per_elem
    if method == "tp":
        parts = {"allreduce": 2 * (D - 1) / D * B * S * h * beta}
    elif method == "tsp" and block == "attn":
        parts = {
            "weight_broadcast": (D - 1) * 4 * h * h / D * beta,
            "kv_allgather": 2 * (D - 1) * B * S * h / D * beta,
        }
    elif method == "tsp" and block == "mlp":
        parts = {"weight_ring": (D - 1) * 3 * I * h / D * beta}
    else:
        raise ValueError(f"unknown method/block {method}/{block}")
    parts["total"] = sum(parts.values())
    return parts


def persistent_bytes(block: str, shape: BlockShape, bytes_per_elem: int) -> int:
    """Per-rank persistent weight-shard bytes (identical for TP and TSP)."""
    params = shape.attn_params if block == "attn" else shape.mlp_params
    return params // shape.world_size * bytes_per_elem


def activation_io_bytes(method: str, shape: BlockShape, bytes_per_elem: int) -> int:
    """Per-rank input + output activation bytes ([B,S,h] replicated for TP, [B,S/D,h] for TSP)."""
    tokens = shape.batch * (shape.seq_len if method == "tp" else shape.local_seq)
    return 2 * tokens * shape.hidden * bytes_per_elem
