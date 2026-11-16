"""Forward-only cuDNN SDPA with native bottom-right masking, without a dense mask."""

from __future__ import annotations

import math
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .config import ConfigError

if TYPE_CHECKING:
    import cudnn

_local = threading.local()
_MAX_PLANS = 32


@dataclass
class _Plan:
    graph: cudnn.pygraph
    tensors: tuple[cudnn.tensor, cudnn.tensor, cudnn.tensor, cudnn.tensor]
    workspace_size: int


class _Context:
    def __init__(self):
        try:
            import cudnn
        except ModuleNotFoundError as exc:
            if exc.name != "cudnn":
                raise
            raise RuntimeError(
                "cudnn_frontend is not supported without nvidia-cudnn-frontend; install the cudnn extra"
            ) from exc
        self.cudnn = cudnn
        self.handle = cudnn.create_handle()
        self._cleanup = weakref.finalize(self, cudnn.destroy_handle, self.handle)
        self.plans: OrderedDict[tuple, _Plan] = OrderedDict()

    def plan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> _Plan:
        key = (q.dtype, *((tuple(t.shape), t.stride()) for t in (q, k, v)))
        plan = self.plans.pop(key, None)
        if plan is None:
            plan = self._build(q, k, v)
        self.plans[key] = plan
        if len(self.plans) > _MAX_PLANS:
            self.plans.popitem(last=False)
        return plan

    def _build(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> _Plan:
        cudnn = self.cudnn
        dtype = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
        graph = cudnn.pygraph(
            io_data_type=dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        qt, kt, vt = (graph.tensor_like(t.transpose(1, 2)) for t in (q, k, v))
        out, _ = graph.sdpa(
            q=qt, k=kt, v=vt,
            attn_scale=1.0 / math.sqrt(q.size(-1)),
            generate_stats=False,
            diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT,
            diagonal_band_right_bound=0,
        )
        batch, length, heads, dim = q.shape
        out.set_output(True).set_dim((batch, heads, length, dim)).set_stride(
            (length * heads * dim, dim, heads * dim, 1)
        )
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
        return _Plan(graph, (qt, kt, vt, out), graph.get_workspace_size())


def cudnn_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply bottom-right causal SDPA to [B, L, H, d] CUDA FP16/BF16 tensors.

    Cached plans contain metadata only. Outputs and workspaces are allocated on the caller's current
    stream for every invocation, so their lifetimes are included in allocator peak measurements.
    """
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("cudnn_frontend is not supported for this device/dtype; requires CUDA FP16/BF16")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ConfigError("cuDNN attention inputs must share a device and dtype")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ConfigError("cuDNN attention expects [B, L, H, d] inputs")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ConfigError("cuDNN attention requires matching batch, heads, head dimension, and K/V lengths")
    if q.size(1) < 1 or k.size(1) < q.size(1):
        raise ConfigError("cuDNN attention requires 0 < query length <= key length")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("cudnn_frontend is forward-only; use inference_mode or no_grad")
    with torch.cuda.device(q.device):
        contexts = getattr(_local, "contexts", None)
        if contexts is None:
            contexts = _local.contexts = {}
        device = q.get_device()
        if device not in contexts:
            contexts[device] = _Context()
        context = contexts[device]
        plan = context.plan(q, k, v)
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        workspace = torch.empty(plan.workspace_size, dtype=torch.uint8, device=q.device)
        context.cudnn.set_stream(context.handle, torch.cuda.current_stream(q.device).cuda_stream)
        qt, kt, vt, out = plan.tensors
        plan.graph.execute({qt: q, kt: k, vt: v, out: output}, workspace, handle=context.handle)
    return output
