"""Shared shape/layout contracts (spec section 3) and run options.

Notation: B batch, S global sequence length, h hidden width, H heads, d = h/H head dim,
I gated-MLP intermediate width, D parallel degree (world size).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

DTYPES: dict[str, torch.dtype] = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

METHODS = ("tp", "tsp")
BLOCKS = ("attn", "mlp")
# "sdpa" lets PyTorch pick; "flash"/"efficient"/"math"/"cudnn" restrict SDPA to one backend;
# "flash_attn" uses the optional Dao-AILab package; "naive" is explicit matmul-softmax.
ATTN_BACKENDS = ("sdpa", "flash", "efficient", "math", "cudnn", "flash_attn", "naive")


class ConfigError(ValueError):
    """Raised for configurations that violate the shared shape contracts."""


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError:
        raise ConfigError(f"unknown dtype {name!r}; expected one of {sorted(DTYPES)}") from None


def dtype_name(dtype: torch.dtype) -> str:
    for name, dt in DTYPES.items():
        if dt == dtype:
            return name
    raise ConfigError(f"unsupported dtype {dtype}")


@dataclass(frozen=True)
class BlockShape:
    """Global operator shape plus the parallel degree D it will be split over."""

    batch: int
    seq_len: int
    hidden: int
    n_heads: int
    intermediate: int
    world_size: int = 2

    def __post_init__(self) -> None:
        for name in ("batch", "seq_len", "hidden", "n_heads", "intermediate", "world_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f"{name} must be a positive int, got {value!r}")
        D = self.world_size
        if self.hidden % self.n_heads:
            raise ConfigError(f"hidden={self.hidden} not divisible by n_heads={self.n_heads}")
        if self.n_heads % D:
            raise ConfigError(f"n_heads={self.n_heads} not divisible by D={D}")
        if self.intermediate % D:
            raise ConfigError(f"intermediate={self.intermediate} not divisible by D={D}")
        if self.seq_len % (2 * D):
            raise ConfigError(
                f"seq_len={self.seq_len} not divisible by 2D={2 * D} (equal zigzag chunks required)"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def heads_per_rank(self) -> int:
        return self.n_heads // self.world_size

    @property
    def hidden_per_rank(self) -> int:
        """Width h/D of one rank's head shard (heads_per_rank * head_dim)."""
        return self.hidden // self.world_size

    @property
    def inter_per_rank(self) -> int:
        return self.intermediate // self.world_size

    @property
    def local_seq(self) -> int:
        """S_loc = S/D tokens held by one TSP rank."""
        return self.seq_len // self.world_size

    @property
    def chunk_len(self) -> int:
        """S_c = S/(2D), length of one zigzag chunk."""
        return self.seq_len // (2 * self.world_size)

    @property
    def attn_params(self) -> int:
        """W_Q, W_K, W_V, W_O (bias-free MHA)."""
        return 4 * self.hidden * self.hidden

    @property
    def mlp_params(self) -> int:
        """W_gate, W_up, W_down."""
        return 3 * self.hidden * self.intermediate

    def check_head_bucket(self, head_bucket: int | None) -> int:
        """Resolve a head-bucket size (None = all local heads) and validate divisibility."""
        hs = self.heads_per_rank
        if head_bucket is None:
            return hs
        if not isinstance(head_bucket, int) or head_bucket < 1:
            raise ConfigError(f"head_bucket must be a positive int, got {head_bucket!r}")
        if hs % head_bucket:
            raise ConfigError(f"heads_per_rank={hs} not divisible by head_bucket={head_bucket}")
        return head_bucket

    def to_dict(self) -> dict:
        return asdict(self)


def check_attn_backend(name: str) -> str:
    if name not in ATTN_BACKENDS:
        raise ConfigError(f"unknown attention backend {name!r}; expected one of {ATTN_BACKENDS}")
    return name
