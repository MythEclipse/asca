"""Decode dengan CUDA Graph: satu replay = satu token.

Kenapa perlu: jalur ``forward_step`` ASCA menjalankan ~2.900 kernel per token pada
preset tiny. Dengan kernel sekecil itu, latensi didominasi *launch overhead* CPU
(~3 µs per launch), bukan GPU. Merekam satu langkah decode ke CUDA graph menghapus
hampir seluruh overhead itu: setelah capture, satu token = satu ``graph.replay()``.

Syarat yang harus dipenuhi (dan sudah dipenuhi di ``asca`` maupun baseline):

* semua bentuk tensor tetap — cache dialokasikan sekali di ``init_cache``;
* tidak ada cabang data-dependent — posisi/window memakai ``scatter_``/``index_select``
  dengan indeks dari tensor, dan ringkasan segmen memakai online softmax;
* tidak ada ``.item()`` / sinkronisasi CPU-GPU di dalam langkah.

Pemakaian::

    dec = GraphDecoder(model, batch=1)
    dec.reset()                       # mulai dari konteks kosong
    logits = dec.step(tokens)         # tokens: (batch, 1) int64
"""

from __future__ import annotations

import torch


class GraphDecoder:
    def __init__(
        self,
        model: torch.nn.Module,
        batch: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        warmup: int = 3,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GraphDecoder membutuhkan CUDA")
        self.model = model.eval()
        p = next(self.model.parameters())
        self.device = device or p.device
        self.dtype = dtype or p.dtype
        self.batch = batch
        self.vocab = self.model.cfg.vocab_size

        # buffer input statis (satu-satunya bagian yang diubah dari luar graph)
        self.idx_buf = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self._cache = None
        self._graph = None
        self._logits = None
        self._steps = 0

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Mulai konteks baru: cache bersih, lalu (re)capture graph."""
        self._cache = self.model.init_cache(self.batch, self.device, self.dtype)
        self.idx_buf.zero_()
        self._steps = 0
        self._capture()

    def _capture(self) -> None:
        torch.cuda.synchronize()
        # warmup di luar graph: alokasi, autotune kernel, isi cache awal.
        # cache di-reset lagi setelah warmup supaya graph mulai dari posisi 0.
        with torch.no_grad():
            for _ in range(3):
                self.model.forward_step(self.idx_buf, self._cache)
        torch.cuda.synchronize()
        self._cache = self.model.init_cache(self.batch, self.device, self.dtype)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._logits = self.model.forward_step(self.idx_buf, self._cache)

    # ------------------------------------------------------------------- pakai
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Satu token per batch: ``tokens`` berbentuk ``(batch, 1)`` int64."""
        if tokens.shape != self.idx_buf.shape:
            raise ValueError(f"bentuk token {tuple(tokens.shape)} != {tuple(self.idx_buf.shape)}")
        self.idx_buf.copy_(tokens)
        self._graph.replay()
        self._steps += 1
        # kembalikan salinan: buffer logits ditimpa pada replay berikutnya
        return self._logits.clone()

    @property
    def cache(self) -> dict:
        return self._cache

    @property
    def n_steps(self) -> int:
        return self._steps
