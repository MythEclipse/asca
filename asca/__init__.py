"""ASCA — Adaptive Sparse Cognitive Architecture.

Riset arsitektur LLM dari nol. Lihat ``docs/SPEC_ASCA.md`` untuk spesifikasi
matematis dan ``docs/EXPERIMENTS.md`` untuk hasil benchmark.
"""

from .block import ASCABlock
from .config import ABLATIONS, ASCAConfig, ablation
from .experts import AdaptiveSparseMoE, SwiGLU
from .memory import SegmentMemory
from .model import ASCAModel, SharedDecoder
from .norm import RMSNorm, apply_rope, rope_cache
from .router import AdaptiveRouter

__all__ = [
    "ABLATIONS",
    "ASCAConfig",
    "ASCABlock",
    "ASCAModel",
    "AdaptiveRouter",
    "AdaptiveSparseMoE",
    "RMSNorm",
    "SegmentMemory",
    "SharedDecoder",
    "SwiGLU",
    "ablation",
    "apply_rope",
    "rope_cache",
]

__version__ = "0.1.0"
