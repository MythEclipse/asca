"""Semantic Encoder (docs/SPEC_ASCA.md §3.1).

Agregasi semantik lokal yang murah (konvolusi kausal depthwise + bottleneck MLP)
dijalankan sekali setelah embedding, supaya blok berat tidak dipakai untuk
membangun bigram/trigram.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ASCAConfig
from .norm import RMSNorm


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.kernel = cfg.sem_conv_kernel
        self.norm = RMSNorm(d)
        self.conv = nn.Conv1d(d, d, kernel_size=self.kernel, groups=d, bias=False)
        hidden = max(8, d // cfg.sem_reduction)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm(x)
        # konvolusi kausal: pad kiri sebanyak kernel-1
        c = F.pad(u.transpose(1, 2), (self.kernel - 1, 0))
        c = self.conv(c).transpose(1, 2)
        return x + self.w2(self.act(self.w1(c)))

    # ------------------------------------------------------------- prefill
    def forward_with_state(self, x: torch.Tensor):
        """Seperti ``forward`` tetapi juga mengembalikan buffer konvolusi decode."""
        u = self.norm(x)
        c = F.pad(u.transpose(1, 2), (self.kernel - 1, 0))
        c = self.conv(c).transpose(1, 2)
        B, T, _ = x.shape
        k = self.kernel
        buf = torch.zeros(B, k, self.d, device=x.device, dtype=x.dtype)
        n_fill = min(T, k)
        positions = torch.arange(T - n_fill, T, device=x.device)
        buf[:, positions % k] = u[:, -n_fill:]
        return x + self.w2(self.act(self.w1(c))), {"buf": buf}

    # ------------------------------------------------------------------ step
    def init_state(self, batch: int, device, dtype) -> dict:
        """Ring buffer ``kernel`` input ternormalisasi terakhir (graph-safe)."""
        return {
            "buf": torch.zeros(batch, self.kernel, self.d, device=device, dtype=dtype),
        }

    def step(self, x_t: torch.Tensor, pos_t: torch.Tensor, rope, state: dict) -> torch.Tensor:
        B = x_t.shape[0]
        u = self.norm(x_t)  # (B,1,D)
        k = self.kernel
        slot = (pos_t % k).view(1, 1, 1).expand(B, 1, self.d)
        state["buf"].scatter_(1, slot, u)  # buf (B,K,D), index (B,1,D)
        j = torch.arange(k, device=x_t.device)
        absolute = pos_t + 1 - k + j
        idx = absolute % k
        buf = state["buf"].index_select(1, idx)  # (B,K,D) urut kronologis
        # slot yang belum terisi diperlakukan sebagai nol (sama dengan zero padding)
        buf = buf * (absolute >= 0).view(1, k, 1)
        c = (buf * self.conv.weight[:, 0, :].transpose(0, 1)).sum(dim=1, keepdim=True)
        return x_t + self.w2(self.act(self.w1(c)))
