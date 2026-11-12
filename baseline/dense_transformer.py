"""Baseline kontrol: Dense Causal Transformer (Llama/GPT-style).

Implementasi sengaja dipisah dari ASCA (bukan sekadar "ASCA dengan flag off")
supaya perbandingan benar-benar head-to-head terhadap arsitektur konvensional:

  * RMSNorm pre-norm
  * full causal self-attention dengan RoPE
  * SwiGLU MLP
  * weight tying (opsional)

Antarmuka identik dengan ``asca.model.ASCAModel``: ``forward(idx, targets)``
mengembalikan ``(logits, loss)`` dan menyediakan ``aux_loss`` (selalu 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from asca.norm import RMSNorm, apply_rope, rope_cache


@dataclass
class DenseConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: Optional[int] = None
    max_seq_len: int = 1024
    d_ff: Optional[int] = None
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    bias: bool = False
    dropout: float = 0.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        if self.d_ff is None:
            self.d_ff = int(8 * self.d_model / 3)
            self.d_ff = (self.d_ff + 7) // 8 * 8


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DenseConfig) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = apply_rope(q.view(B, T, self.h, self.dh).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.h, self.dh).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseBlock(nn.Module):
    def __init__(self, cfg: DenseConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        return x + self.mlp(self.norm_mlp(x))


class DenseTransformer(nn.Module):
    def __init__(self, cfg: DenseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.tied = cfg.tie_embeddings
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w2.weight"):
                with torch.no_grad():
                    p.mul_(scale)
        self.last_aux_loss = torch.zeros(())

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"panjang sekuens {T} melebihi max_seq_len {self.cfg.max_seq_len}")
        x = self.embed(idx)
        cos, sin = rope_cache(T, self.cfg.head_dim, self.cfg.rope_theta, idx.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight) if self.tied else self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.last_aux_loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def active_params_per_token(self) -> int:
        return self.num_params(non_embedding=True)
