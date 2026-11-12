"""Konfigurasi ASCA.

Satu dataclass mengendalikan seluruh komponen, sehingga ablasi (matriks A0..A6
pada docs/SPEC_ASCA.md §5) cukup dilakukan lewat konfigurasi.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ASCAConfig:
    # --- dasar ---
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: Optional[int] = None
    max_seq_len: int = 1024
    bias: bool = False
    dropout: float = 0.0
    init_std: float = 0.02

    # --- 3.1 Semantic Encoder ---
    use_semantic_encoder: bool = True
    sem_conv_kernel: int = 4
    sem_reduction: int = 4

    # --- 3.2 Adaptive Hybrid Attention ---
    use_hybrid_attention: bool = True
    local_window: int = 128  # >= max_seq_len berarti attention kausal penuh
    linear_chunk: int = 64
    rope_theta: float = 10000.0

    # --- 3.3 Adaptive Sparse MoE ---
    use_moe: bool = True
    n_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 1
    expert_ff: Optional[int] = None
    shared_ff: Optional[int] = None
    router_threshold: float = 0.0  # 0 = adaptive skip dimatikan
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.001
    dense_ff: Optional[int] = None  # dipakai saat use_moe=False

    # --- 3.4 Efficient Memory ---
    use_memory: bool = True
    n_memory_slots: int = 32
    mem_segment_len: int = 256

    # --- 3.5 Shared Decoder ---
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model harus habis dibagi n_heads")
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError("head_dim * n_heads harus sama dengan d_model")
        if self.expert_ff is None:
            self.expert_ff = max(32, self.d_model // 2)
        if self.shared_ff is None:
            self.shared_ff = 2 * self.d_model
        if self.dense_ff is None:
            self.dense_ff = _round_up(int(8 * self.d_model / 3), 8)
        if self.top_k > self.n_experts:
            raise ValueError("top_k tidak boleh melebihi n_experts")

    def to_dict(self) -> dict:
        return asdict(self)


def _round_up(n: int, m: int) -> int:
    return int((n + m - 1) // m * m)


#: Matriks ablasi (docs/SPEC_ASCA.md §5). Nama -> override terhadap konfigurasi penuh.
ABLATIONS: dict[str, dict] = {
    "A0": {},  # ASCA penuh
    "A1": {"use_semantic_encoder": False},
    "A2": {"use_hybrid_attention": False},
    "A3": {"use_moe": False},
    "A4": {"use_memory": False},
    "A5": {"n_shared_experts": 0},
    "A6": {"router_threshold": 0.0},
}


def ablation(base: ASCAConfig, name: str) -> ASCAConfig:
    """Kembalikan salinan `base` dengan override ablasi `name` (mis. ``"A2"``)."""
    if name not in ABLATIONS:
        raise KeyError(f"ablasi tidak dikenal: {name!r} (pilihan: {sorted(ABLATIONS)})")
    cfg = copy.deepcopy(base)
    for k, v in ABLATIONS[name].items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg
