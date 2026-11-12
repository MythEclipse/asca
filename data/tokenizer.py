"""Tokenizer: BPE (via `tokenizers`) dengan fallback byte-level tanpa dependensi.

Dipakai untuk Tahap 1 (tokenizer & representasi bahasa). Keduanya mengekspos
antarmuka yang sama: ``vocab_size``, ``encode(str) -> list[int]``,
``decode(list[int]) -> str``, ``eos_id``, ``pad_id``.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence


class ByteTokenizer:
    """Tokenizer byte-level murni (vocab 256). Selalu tersedia, tanpa dependensi."""

    pad_id = 0
    eos_id = 0

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("byte\n")

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        return cls()


class BPETokenizer:
    """Wrapper tipis di atas ``tokenizers.Tokenizer`` (BPE byte-level)."""

    def __init__(self, tok, pad_id: int = 0, eos_id: int = 1) -> None:
        self._tok = tok
        self.pad_id = pad_id
        self.eos_id = eos_id

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=True)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


def train_bpe(
    files: Iterable[str],
    vocab_size: int = 4096,
    out_path: str = "artifacts/tokenizer.json",
    special_tokens: Sequence[str] = ("<pad>", "<eos>"),
) -> BPETokenizer:
    """Latih BPE byte-level pada file teks lokal."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(f) for f in files], trainer)
    wrapper = BPETokenizer(tok)
    wrapper.save(out_path)
    return wrapper


def load_tokenizer(path: str | None):
    """Muat tokenizer dari ``path``; ``None`` -> :class:`ByteTokenizer`."""
    if path and os.path.exists(path):
        return BPETokenizer.load(path)
    return ByteTokenizer()
