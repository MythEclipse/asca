"""Sparse Experts + Shared Expert (docs/SPEC_ASCA.md §3.3).

``AdaptiveSparseMoE`` = shared expert (selalu aktif) + top-k routed experts
dengan adaptive compute (threshold skip). Token diproses per-expert lewat
gather/scatter sehingga FLOPs sebanding jumlah token yang benar-benar dirutekan,
bukan ``E`` kali semua token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ASCAConfig
from .router import AdaptiveRouter


class SwiGLU(nn.Module):
    """MLP SwiGLU: ``W2(silu(W1 x) * W3 x)``."""

    def __init__(self, d_in: int, d_hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaptiveSparseMoE(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = AdaptiveRouter(cfg)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_ff, cfg.bias) for _ in range(cfg.n_experts)]
        )
        self.shared = (
            SwiGLU(cfg.d_model, cfg.shared_ff, cfg.bias) if cfg.n_shared_experts > 0 else None
        )
        # statistik untuk logging (tidak mempengaruhi gradien)
        self.register_buffer("_expert_counts", torch.zeros(cfg.n_experts), persistent=False)
        self.register_buffer("_tokens_seen", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)
        weights, top_idx = self.router.route(logits)
        aux = self.router.aux_loss(logits, top_idx)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = top_idx == e  # (N, k)
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            h = expert(flat.index_select(0, tok))
            w = weights[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, h * w)

        if self.shared is not None:
            out = out + self.shared(flat)

        if not self.training:
            with torch.no_grad():
                self._expert_counts += F.one_hot(top_idx, self.n_experts).sum(dim=(0, 1)).float()
                self._tokens_seen += flat.shape[0]

        return out.view(B, T, D), aux

    @torch.no_grad()
    def expert_usage(self) -> torch.Tensor:
        """Fraksi token yang dirutekan ke setiap expert sejak reset terakhir."""
        total = self._expert_counts.sum().clamp(min=1.0)
        return self._expert_counts / total

    @torch.no_grad()
    def reset_usage(self) -> None:
        self._expert_counts.zero_()
        self._tokens_seen.zero_()
