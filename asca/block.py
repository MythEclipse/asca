"""ASCA block: pre-norm residual atas attention, memory, dan MoE."""

from __future__ import annotations

import torch
from torch import nn

from .attention import AdaptiveHybridAttention
from .config import ASCAConfig
from .experts import AdaptiveSparseMoE, SwiGLU
from .memory import SegmentMemory
from .norm import RMSNorm


class ASCABlock(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.use_memory = cfg.use_memory
        self.use_moe = cfg.use_moe

        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = AdaptiveHybridAttention(cfg)

        if self.use_memory:
            self.norm_mem = RMSNorm(cfg.d_model)
            self.mem = SegmentMemory(cfg)

        self.norm_mlp = RMSNorm(cfg.d_model)
        if self.use_moe:
            self.mlp = AdaptiveSparseMoE(cfg)
        else:
            self.mlp = SwiGLU(cfg.d_model, cfg.dense_ff, cfg.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm_attn(x), cos, sin)

        if self.use_memory:
            x = x + self.mem(self.norm_mem(x))

        aux = x.new_zeros(())
        if self.use_moe:
            h, aux = self.mlp(self.norm_mlp(x))
            x = x + h
        else:
            x = x + self.mlp(self.norm_mlp(x))
        return x, aux

    # ------------------------------------------------------------- prefill
    def forward_with_state(self, x: torch.Tensor, cos=None, sin=None):
        """Seperti ``forward`` tetapi juga mengembalikan state cache decode."""
        if hasattr(self.attn, "forward_with_state"):
            a, attn_state = self.attn.forward_with_state(self.norm_attn(x), cos, sin)
        else:
            a, attn_state = self.attn(self.norm_attn(x), cos, sin), None
        x = x + a
        state = {"attn": attn_state}
        if self.use_memory:
            m, mem_state = self.mem.forward_with_state(self.norm_mem(x))
            x = x + m
            state["mem"] = mem_state
        aux = x.new_zeros(())
        if self.use_moe:
            h, aux = self.mlp(self.norm_mlp(x))
            x = x + h
        else:
            x = x + self.mlp(self.norm_mlp(x))
        return x, aux, state

    # ------------------------------------------------------------------ step
    def init_state(self, batch: int, device, dtype) -> dict:
        st = {"attn": self.attn.init_state(batch, device, dtype)}
        if self.use_memory:
            st["mem"] = self.mem.init_state(batch, device, dtype)
        return st

    def step(self, x_t: torch.Tensor, pos_t: torch.Tensor, rope, state: dict):
        """Satu token (decode inkremental)."""
        x = x_t + self.attn.step(self.norm_attn(x_t), pos_t, rope, state["attn"])
        if self.use_memory:
            x = x + self.mem.step(self.norm_mem(x), pos_t, rope, state["mem"])
        aux = x.new_zeros(())
        if self.use_moe:
            h, aux = self.mlp(self.norm_mlp(x))
            x = x + h
        else:
            x = x + self.mlp(self.norm_mlp(x))
        return x, aux
