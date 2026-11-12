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
        self.cfg_theta = cfg.rope_theta
        self.max_len = cfg.max_seq_len
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

    # --- decode inkremental: KV-cache penuh, dialokasikan di awal (graph-safe) ---
    def init_state(self, batch: int, device, dtype) -> dict:
        return {
            "k": torch.zeros(batch, self.h, self.max_len, self.dh, device=device, dtype=dtype),
            "v": torch.zeros(batch, self.h, self.max_len, self.dh, device=device, dtype=dtype),
        }

    def step(self, x_t: torch.Tensor, pos_t: torch.Tensor, rope, state: dict) -> torch.Tensor:
        B, _, D = x_t.shape
        q, k, v = self.qkv(x_t).split(D, dim=-1)
        q = q.view(B, 1, self.h, self.dh).transpose(1, 2)
        k = k.view(B, 1, self.h, self.dh).transpose(1, 2)
        v = v.view(B, 1, self.h, self.dh).transpose(1, 2)
        cos_all, sin_all = rope
        cos = cos_all.index_select(0, pos_t)
        sin = sin_all.index_select(0, pos_t)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        slot = pos_t.view(1, 1, 1, 1).expand(B, self.h, 1, self.dh)
        state["k"].scatter_(2, slot, k)
        state["v"].scatter_(2, slot, v)
        j = torch.arange(self.max_len, device=x_t.device)
        valid = j <= pos_t
        scores = (q @ state["k"].transpose(-1, -2)) / math.sqrt(self.dh)
        scores = scores.masked_fill(~valid.view(1, 1, 1, -1), -1e30)
        out = torch.softmax(scores, dim=-1) @ state["v"]
        return self.proj(out.transpose(1, 2).reshape(B, 1, D))


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

    def init_state(self, batch: int, device, dtype) -> dict:
        return self.attn.init_state(batch, device, dtype)

    def step(self, x_t: torch.Tensor, pos_t: torch.Tensor, rope, state: dict) -> torch.Tensor:
        x = x_t + self.attn.step(self.norm_attn(x_t), pos_t, rope, state)
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

    # --------------------------------------------------------------- decode
    def init_cache(self, batch: int = 1, device=None, dtype=None) -> dict:
        p = next(self.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        cos, sin = rope_cache(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype)
        return {
            "pos": torch.zeros((), dtype=torch.long, device=device),
            "n": 0,
            "rope": (cos, sin),
            "layers": [b.init_state(batch, device, dtype) for b in self.blocks],
        }

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor):
        """Prompt paralel + KV-cache penuh yang sudah terisi (pos = T)."""
        self.eval()
        B, T = idx.shape
        logits, _ = self(idx)
        device = idx.device
        dtype = self.embed.weight.dtype
        cos_all, sin_all = rope_cache(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype)
        cos, sin = cos_all[:T], sin_all[:T]
        x = self.embed(idx)
        layers = []
        for blk in self.blocks:
            u = blk.norm_attn(x)
            q, k, v = blk.attn.qkv(u).split(self.cfg.d_model, dim=-1)
            k = apply_rope(k.view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2), cos, sin)
            v = v.view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
            st = {
                "k": torch.zeros(B, self.cfg.n_heads, self.cfg.max_seq_len, self.cfg.head_dim, device=device, dtype=dtype),
                "v": torch.zeros(B, self.cfg.n_heads, self.cfg.max_seq_len, self.cfg.head_dim, device=device, dtype=dtype),
            }
            st["k"][:, :, :T] = k
            st["v"][:, :, :T] = v
            layers.append(st)
            x = x + blk.attn(u, cos, sin)
            x = x + blk.mlp(blk.norm_mlp(x))
        return logits, {
            "pos": torch.tensor(T, dtype=torch.long, device=device),
            "n": T,
            "rope": (cos_all, sin_all),
            "layers": layers,
        }

    def forward_step(self, idx_t: torch.Tensor, cache: dict) -> torch.Tensor:
        if idx_t.shape[1] != 1:
            raise ValueError("forward_step hanya menerima satu token per panggilan")
        if cache["n"] >= self.cfg.max_seq_len:
            raise ValueError("melebihi max_seq_len; reset cache")
        pos_t = cache["pos"]
        x = self.embed(idx_t)
        for block, st in zip(self.blocks, cache["layers"]):
            x = block.step(x, pos_t, cache["rope"], st)
        x = self.norm_f(x)
        cache["pos"] += 1
        cache["n"] += 1
        return F.linear(x, self.embed.weight) if self.tied else self.head(x)

    @torch.no_grad()
    def generate_cached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        cache = self.init_cache(idx.shape[0], idx.device, self.embed.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits = self.forward_step(idx[:, t : t + 1], cache)
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, step_logits.size(-1))
                thresh = torch.topk(step_logits, k, dim=-1).values[:, -1:]
                step_logits = step_logits.masked_fill(step_logits < thresh, float("-inf"))
            nxt = torch.multinomial(F.softmax(step_logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            logits = self.forward_step(nxt, cache)
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def active_params_per_token(self) -> int:
        return self.num_params(non_embedding=True)
