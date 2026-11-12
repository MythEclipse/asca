"""Model ASCA lengkap (docs/SPEC_ASCA.md §2, §3.5)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import ASCABlock
from .config import ASCAConfig
from .encoder import SemanticEncoder
from .norm import RMSNorm, rope_cache


class SharedDecoder(nn.Module):
    """Norm akhir + gated semantic skip dari embedding + proyeksi (opsional tied)."""

    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.tied = cfg.tie_embeddings
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, h: torch.Tensor, h0: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        z = self.norm(h) + torch.sigmoid(self.gate) * h0
        if self.tied:
            return F.linear(z, embed_weight)
        return self.head(z)


class ASCAModel(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.sem = SemanticEncoder(cfg) if cfg.use_semantic_encoder else nn.Identity()
        self.blocks = nn.ModuleList([ASCABlock(cfg) for _ in range(cfg.n_layers)])
        self.decoder = SharedDecoder(cfg)
        self.apply(self._init_weights)
        self._rescale_residual_projections()
        self.last_aux_loss: torch.Tensor = torch.zeros(())

    # ------------------------------------------------------------------ init
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _rescale_residual_projections(self) -> None:
        """Skala output proyeksi residual dengan 1/sqrt(2L) (GPT-2 style)."""
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w2.weight"):
                with torch.no_grad():
                    param.mul_(scale)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"panjang sekuens {T} melebihi max_seq_len {self.cfg.max_seq_len}")

        h0 = self.embed(idx) * math.sqrt(self.cfg.d_model)
        h = self.sem(h0)
        cos, sin = rope_cache(T, self.cfg.head_dim, self.cfg.rope_theta, idx.device, h.dtype)

        aux_total = h.new_zeros(())
        for block in self.blocks:
            h, aux = block(h, cos, sin)
            aux_total = aux_total + aux
        self.last_aux_loss = aux_total / max(1, len(self.blocks))

        logits = self.decoder(h, h0, self.embed.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @property
    def aux_loss(self) -> torch.Tensor:
        """Rata-rata load-balancing loss seluruh blok dari forward terakhir."""
        return self.last_aux_loss

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Decode autoregresif tanpa KV-cache (recompute penuh; cukup untuk prototipe)."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    # --------------------------------------------------------------- statistik
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def active_params_per_token(self) -> int:
        """Perkiraan parameter yang benar-benar dipakai per token (MoE sparse).

        Embedding dikeluarkan dari hitungan (lookup, bukan matmul). Perhitungan
        FLOPs yang otoritatif ada di ``benchmarks/bench_flops.py``.
        """
        n = sum(p.numel() for p in self.parameters())
        n -= self.embed.weight.numel()
        for block in self.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                total = sum(p.numel() for p in mlp.experts.parameters())
                per_expert = total / max(1, len(mlp.experts))
                n -= int(total - per_expert * self.cfg.top_k)
        return n
