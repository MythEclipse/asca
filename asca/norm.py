"""Normalisasi dan positional encoding yang dipakai bersama ASCA & baseline."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019) — dihitung dalam fp32 untuk stabilitas."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return h.to(dtype) * self.weight


def rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kembalikan ``(cos, sin)`` berbentuk ``(seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Terapkan RoPE pada ``x`` berbentuk ``(B, H, T, Dh)``.

    ``cos``/``sin`` berbentuk ``(T, Dh // 2)``.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)
