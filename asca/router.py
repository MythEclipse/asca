"""Adaptive Router (docs/SPEC_ASCA.md §3.3).

Router menghasilkan logit per expert dan menghitung auxiliary loss
(switch load-balance + z-loss). Keputusan *adaptive compute* (threshold skip)
dilakukan di ``experts.AdaptiveSparseMoE`` karena butuh bobot hasil top-k.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ASCAConfig


class AdaptiveRouter(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.threshold = cfg.router_threshold
        self.aux_coef = cfg.aux_loss_coef
        self.z_coef = cfg.z_loss_coef
        self.proj = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, D)`` -> logits ``(N, E)``."""
        return self.proj(x)

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Kembalikan ``(weights, top_idx)`` berbentuk ``(N, k)``.

        Bobot dinormalisasi ulang atas top-k. Bila ``threshold > 0``, token yang
        ``max_e p_e < threshold`` mendapat bobot nol (hanya shared expert yang jalan).
        """
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        if self.threshold > 0.0:
            keep = (top_w >= self.threshold).to(top_w.dtype)
        else:
            keep = torch.ones_like(top_w)
        weights = top_w * keep
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return weights, top_i

    def aux_loss(self, logits: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Switch load-balancing loss + router z-loss."""
        n = logits.shape[0]
        if n == 0:
            return logits.new_zeros(())
        probs = F.softmax(logits, dim=-1)
        selected = F.one_hot(top_idx, self.n_experts).sum(dim=1).to(probs.dtype)  # (N,E)
        frac = selected.mean(dim=0)  # f_e
        mean_p = probs.mean(dim=0)  # p_e
        switch = self.n_experts * (frac * mean_p).sum()
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean().to(logits.dtype)
        return self.aux_coef * switch + self.z_coef * z
