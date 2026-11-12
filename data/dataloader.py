"""Korpus dan dataloader streaming.

Sumber korpus (urutan prioritas sesuai argumen ``kind``):
  * ``"synthetic"`` — korpus sintetis terstruktur (Markov + long-range copy).
    Selalu tersedia tanpa jaringan; dipakai untuk smoke test & unit test.
  * ``"file"``      — file teks/JSONL lokal.
  * ``"hf"``        — Hugging Face: ``datasets`` streaming bila terpasang,
    jika tidak ``huggingface_hub.hf_hub_download`` untuk satu file.

Token disimpan sebagai memmap ``.npy`` di ``data/cache/`` supaya training bisa
dilanjutkan tanpa tokenisasi ulang.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")


# --------------------------------------------------------------------- korpus
def synthetic_text(n_chars: int = 400_000, seed: int = 0, vocab_words: int = 48) -> str:
    """Korpus sintetis dengan struktur yang bisa dipelajari (bukan acak murni).

    Setiap kata punya tabel bigram tetap, kalimat diakhiri '.', dan sesekali
    disisipkan pasangan ``KEY_i ... KEY_i`` untuk menguji ketergantungan jarak jauh.
    """
    rng = random.Random(seed)
    words = [f"w{i:02d}" for i in range(vocab_words)]
    succ = {w: [rng.choice(words) for _ in range(3)] for w in words}
    parts: list[str] = []
    total = 0
    sent_id = 0
    while total < n_chars:
        sent_id += 1
        n = rng.randint(6, 14)
        cur = rng.choice(words)
        sent = [cur]
        for _ in range(n):
            cur = rng.choice(succ[cur])
            sent.append(cur)
        if sent_id % 7 == 0:
            key = f"KEY_{rng.randint(0, 99):02d}"
            sent = [key] + sent + [key]
        text = " ".join(sent) + " .\n"
        parts.append(text)
        total += len(text)
    return "".join(parts)


def _read_text_file(path: str) -> str:
    if path.endswith(".jsonl"):
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
        return "\n".join(chunks)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def fetch_text(
    kind: str = "synthetic",
    path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    max_chars: int = 2_000_000,
    seed: int = 0,
) -> str:
    if kind == "synthetic":
        return synthetic_text(max_chars, seed=seed)

    if kind == "file":
        if not path:
            raise ValueError("kind='file' butuh path")
        paths = [path] if os.path.isfile(path) else [
            os.path.join(path, f) for f in sorted(os.listdir(path))
        ]
        text = "\n".join(_read_text_file(p) for p in paths)
        return text[:max_chars]

    if kind == "hf":
        if not hf_repo:
            raise ValueError("kind='hf' butuh hf_repo")
        try:  # jalur utama: streaming, tanpa mengunduh seluruh dataset
            from datasets import load_dataset

            ds = load_dataset(hf_repo, split="train", streaming=True)
            out: list[str] = []
            total = 0
            for row in ds:
                t = row.get("text") or row.get("content") or ""
                out.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            return "".join(out)[:max_chars]
        except ImportError:
            from huggingface_hub import hf_hub_download

            if not hf_file:
                raise RuntimeError(
                    "`datasets` tidak terpasang dan hf_file tidak diberikan. "
                    "Pasang `datasets` ke venv bersama, atau berikan hf_file "
                    "(mis. 'sample/10BT/000_00000.parquet' -> gunakan kind='file' setelah unduh)."
                )
            local = hf_hub_download(repo_id=hf_repo, filename=hf_file, repo_type="dataset")
            return _read_text_file(local)[:max_chars]

    raise ValueError(f"kind korpus tidak dikenal: {kind!r}")


# ------------------------------------------------------------------- dataset
@dataclass
class TokenDataset:
    tokens: np.ndarray  # dtype uint16/uint32, seluruh korpus dalam satu array
    val_fraction: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        n_val = max(2, int(len(self.tokens) * self.val_fraction))
        self._n_train = len(self.tokens) - n_val
        self._rng = np.random.default_rng(self.seed)

    @property
    def vocab_size(self) -> int:
        return 256 if self.tokens.dtype == np.uint8 else int(self.tokens.max()) + 1

    def n_tokens(self, split: str = "train") -> int:
        return self._n_train if split == "train" else len(self.tokens) - self._n_train

    def get_batch(self, split: str, batch_size: int, seq_len: int, device=None):
        import torch

        lo, hi = (0, self._n_train) if split == "train" else (self._n_train, len(self.tokens))
        span = hi - lo - seq_len - 1
        if span <= 0:
            raise ValueError("korpus terlalu pendek untuk seq_len yang diminta")
        starts = self._rng.integers(lo, lo + span, size=batch_size)
        x = np.stack([self.tokens[s : s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if device is not None:
            xt, yt = xt.to(device), yt.to(device)
        return xt, yt


def encode_corpus(text: str, tokenizer, batch_chars: int = 200_000) -> np.ndarray:
    """Tokenisasi korpus per potongan agar hemat RAM, hasil digabung."""
    ids: list[int] = []
    for i in range(0, len(text), batch_chars):
        ids.extend(tokenizer.encode(text[i : i + batch_chars]))
    arr = np.asarray(ids, dtype=np.int64)
    return arr.astype(np.uint16 if tokenizer.vocab_size < 65_536 else np.uint32)


def build_dataset(
    kind: str = "synthetic",
    tokenizer=None,
    path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    max_chars: int = 2_000_000,
    val_fraction: float = 0.02,
    seed: int = 0,
    cache: bool = True,
) -> TokenDataset:
    from .tokenizer import ByteTokenizer

    tokenizer = tokenizer or ByteTokenizer()
    tag = hashlib.sha1(
        f"{kind}|{path}|{hf_repo}|{hf_file}|{max_chars}|{seed}|{tokenizer.vocab_size}".encode()
    ).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"tokens_{tag}.npy")

    if cache and os.path.exists(cache_path):
        tokens = np.load(cache_path)
    else:
        text = fetch_text(kind, path, hf_repo, hf_file, max_chars, seed)
        tokens = encode_corpus(text, tokenizer)
        if cache:
            np.save(cache_path, tokens)
    return TokenDataset(tokens=tokens, val_fraction=val_fraction, seed=seed)
