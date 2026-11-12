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

    # --------------------------------------------------------------- decode
    def init_cache(self, batch: int = 1, device=None, dtype=None) -> dict:
        """Inisialisasi cache decode (semua tensor berukuran tetap, graph-safe)."""
        p = next(self.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        cos, sin = rope_cache(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype)
        return {
            "pos": torch.zeros((), dtype=torch.long, device=device),
            "n": 0,  # penghitung python (tanpa .item(), tidak sync)
            "rope": (cos, sin),
            "sem": self.sem.init_state(batch, device, dtype)
            if hasattr(self.sem, "init_state")
            else {},
            "layers": [blk.init_state(batch, device, dtype) for blk in self.blocks],
        }

    def forward_step(self, idx_t: torch.Tensor, cache: dict) -> torch.Tensor:
        """Satu token dengan cache: ``idx_t`` berbentuk ``(B,1)`` -> logits ``(B,1,V)``.

        Identik secara matematis dengan ``forward`` pada sekuens penuh (diverifikasi
        ``tests/run_tests.py``), tetapi biaya per token O(w·D + Dh²) dan tidak ada
        cabang data-dependent sehingga bisa direkam ke CUDA graph.
        """
        if idx_t.shape[1] != 1:
            raise ValueError("forward_step hanya menerima satu token per panggilan")
        if cache["n"] >= self.cfg.max_seq_len:
            raise ValueError("melebihi max_seq_len; reset cache")
        pos_t = cache["pos"]
        rope = cache["rope"]

        h0 = self.embed(idx_t) * math.sqrt(self.cfg.d_model)
        h = self.sem.step(h0, pos_t, rope, cache["sem"]) if hasattr(self.sem, "step") else self.sem(h0)
        aux_total = h.new_zeros(())
        for blk, st in zip(self.blocks, cache["layers"]):
            h, aux = blk.step(h, pos_t, rope, st)
            aux_total = aux_total + aux
        self.last_aux_loss = aux_total / max(1, len(self.blocks))
        cache["pos"] += 1  # in-place: ikut terekam di dalam CUDA graph
        cache["n"] += 1
        return self.decoder(h, h0, self.embed.weight)

    @torch.no_grad()
    def generate_cached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Decode autoregresif memakai cache (satu token per langkah)."""
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
            probs = F.softmax(step_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            logits = self.forward_step(nxt, cache)
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
