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
