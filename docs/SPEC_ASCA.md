# ASCA — Adaptive Sparse Cognitive Architecture

**Spesifikasi Matematis & Dataflow Tensor**
Versi: `0.1-draft` · Status: desain awal, belum divalidasi empiris
Aturan proyek: **tidak ada klaim keunggulan sebelum dibuktikan benchmark terhadap baseline kontrol.**

---

## 1. Notasi

| Simbol | Arti |
|---|---|
| $B$ | batch size |
| $T$ | panjang sekuens |
| $D$ | dimensi model (`d_model`) |
| $H$ | jumlah head, $D_h = D/H$ |
| $V$ | ukuran vocabulary |
| $L$ | jumlah blok (`n_layers`) |
| $E$ | jumlah routed expert |
| $k$ | top-$k$ routing, $k \ll E$ |
| $w$ | lebar sliding window (attention lokal) |
| $C$ | ukuran chunk (attention linear) |
| $M$ | jumlah slot memori |
| $L_s$ | panjang segmen memori |

Semua tensor aktivasi berbentuk `(B, T, D)` kecuali dinyatakan lain.
Indeks posisi $t = 1..T$, head $h = 1..H$.

**Aturan kausalitas (non-negotiable):** untuk setiap komponen, keluaran di posisi $t$
hanya boleh bergantung pada $x_{1..t}$. Ini diverifikasi otomatis oleh
`tests/run_tests.py::test_causality_*` (perturbasi token masa depan tidak boleh
mengubah logit masa lalu). Setiap mekanisme baru wajib lulus tes ini sebelum masuk
ke training.

---

## 2. Pipeline

```
token ids  X ∈ ℕ^{B×T}
   │
   ├─ Embedding  H⁰ = E[X] · √D                    (B,T,D)
   │
   ├─ SemanticEncoder (sekali, murah, kausal)      → H⁰'
   │
   ├─ for l in 1..L:  ASCABlock_l
   │      ├─ AdaptiveHybridAttention  (lokal window w  ⊕  linear global)
   │      ├─ SegmentMemory            (baca ringkasan segmen masa lalu, opsional)
   │      └─ AdaptiveSparseMoE        (shared expert + top-k routed, threshold skip)
   │
   ├─ SharedDecoder:  RMSNorm → gated skip dari embedding → proyeksi tertaut (tied)
   │
   └─ logits ∈ ℝ^{B×T×V}
```

Setiap komponen punya flag di `ASCAConfig`. Dengan semua flag dimatikan, ASCA
tereduksi menjadi Transformer dense kausal — sehingga ablasi bisa dilakukan
murni lewat konfigurasi, tanpa mengubah kode.

---

## 3. Komponen

### 3.1 Semantic Encoder (sekali di input)

Motivasi: blok berat sebaiknya menerima token yang sudah terkontekstualisasi lokal,
sehingga kedalaman jaringan dipakai untuk komputasi semantik tingkat tinggi, bukan
untuk agregasi bigram/trigram.

$$
\begin{aligned}
u &= \mathrm{RMSNorm}(H^0) \\
c_t &= \sum_{j=0}^{K-1} W^{c}_{:,j} \odot u_{t-j} \quad (\text{konvolusi kausal, } K=4,\ \text{zero-pad di kiri}) \\
s_t &= W_2\,\sigma\!\left(W_1 c_t\right), \quad W_1 \in \mathbb{R}^{(D/r)\times D},\; W_2 \in \mathbb{R}^{D\times (D/r)},\; r=4 \\
H^{0\prime}_t &= H^0_t + s_t
\end{aligned}
$$

Biaya: $O(T D^2 / r)$ — jauh di bawah satu blok attention penuh.
Kausal: konvolusi hanya melihat $t-K+1..t$. ✓

### 3.2 Adaptive Hybrid Attention

Dua jalur paralel per head, digabung dengan gate data-dependent:

**Jalur A — local causal attention (window $w$).** Dengan RoPE $\mathcal{R}_{\theta}$:

$$
A^{\text{loc}} = \mathrm{softmax}\!\left(\frac{\mathcal{R}q\,(\mathcal{R}k)^\top}{\sqrt{D_h}} + M^{\text{win}}\right) v,
\qquad M^{\text{win}}_{ij} = \begin{cases}0 & i-w < j \le i\\ -\infty & \text{lainnya}\end{cases}
$$

Biaya $O(T\,w\,D)$.

**Jalur B — gated linear attention (global, $O(T)$).** Feature map
$\varphi(x) = \mathrm{elu}(x)+1 > 0$. State per head $S_t \in \mathbb{R}^{D_h \times D_h}$:

$$
S_t = \lambda_t \odot S_{t-1} + \varphi(k_t)\,v_t^\top, \qquad
o^{\text{glb}}_t = \frac{\varphi(q_t)^\top S_t}{\varphi(q_t)^\top \sum_{j\le t}\varphi(k_j) + \epsilon}
$$

dengan forget gate per head $\lambda_t = \sigma\!\left(W_\lambda\, \bar{x}_{\text{chunk}(t)}\right) \in (0,1)$
(konstan di dalam satu chunk — gating tingkat chunk, bukan token; penyempurnaan ke
gating per token dicatat sebagai future work). Implementasi memakai bentuk chunked
paralel: chunk $c$ berisi posisi $C(c-1)+1 .. Cc$:

$$
o^{\text{intra}} = \left(\varphi(Q_c)\varphi(K_c)^\top \odot \mathrm{tril}\right) V_c,\quad
o^{\text{inter}} = \varphi(Q_c)\, S_{c-1},\quad
S_c = \lambda_c S_{c-1} + \varphi(K_c)^\top V_c
$$

Jumlah iterasi chunk = $T/C$ (mis. 8 untuk $T{=}1024, C{=}128$) — bukan loop per token.
Biaya $O(T C D_h + T D_h^2)$, jadi total sub-kuadratik.

**Chunk adaptif.** Biaya intra-chunk tumbuh $O(TC)$ dan VRAM $O(C^2)$, sementara
loop yang lebih sedikit lebih cepat di GPU. Karena itu implementasi memakai
$C_{\text{eff}} = \max(8, \min(C, \max(8, T/4)))$ supaya sifat sub-kuadratik tidak
hilang pada $T$ kecil. Terukur (preset tiny, batch 4, bf16, fwd+bwd):
$T{=}512$: $C{=}64 \to 4.35$ ms vs $C{=}128 \to 2.57$ ms; $T{=}256$: $C{=}128$
justru lebih boros FLOPs sehingga dibatasi menjadi 64.

**Gate penggabung** (per token, per kanal):

$$
g = \sigma\!\left(W_g x\right) \in (0,1)^{D}, \qquad
o = g \odot o^{\text{loc}} + (1-g) \odot o^{\text{glb}}
$$

Jalur lokal menangani sintaksis/jarak dekat, jalur linear membawa konteks jauh
dengan memori $O(D_h^2)$ per head (bukan $O(T)$ KV-cache).
Ini yang membuat decode panjang tidak meledak VRAM.

### 3.3 Adaptive Sparse MoE + Router

Router (per token $x_t \in \mathbb{R}^D$):

$$
\ell_t = W_r x_t \in \mathbb{R}^{E},\qquad p_t = \mathrm{softmax}(\ell_t),\qquad
\mathcal{T}_t = \mathrm{topk}(p_t, k)
$$

**Adaptive compute (threshold skip).** Dengan $\tau \in [0,1)$:

$$
m_t = \mathbb{1}\!\left[\max_e p_{t,e} \ge \tau\right]
$$

Jika $m_t = 0$, token hanya melewati shared expert (komputasi routed = 0).
Bobot routing dinormalisasi ulang:

$$
\tilde{w}_{t,e} = \frac{p_{t,e}\, m_t}{\sum_{e' \in \mathcal{T}_t} p_{t,e'}}
$$

**Shared expert** (selalu aktif, menyerap pengetahuan umum — mencegah expert
routing kelaparan pada korpus kecil):

$$
y^{\text{sh}}_t = W^{\text{sh}}_2\,\sigma\!\left(W^{\text{sh}}_1 x_t\right)
$$

**Routed experts** (SwiGLU, lebar kecil $d_e$):

$$
y^{\text{rt}}_t = \sum_{e \in \mathcal{T}_t} \tilde{w}_{t,e}\, \mathrm{Expert}_e(x_t)
$$

$$
\mathrm{ASCA\text{-}MoE}(x_t) = y^{\text{sh}}_t + y^{\text{rt}}_t
$$

**Load balancing** (Switch-style, dengan $N = B\cdot T$, $f_e$ = fraksi token,
$\bar{p}_e$ = rata-rata prob):

$$
\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{e=1}^{E} f_e \bar{p}_e
\;+\; \beta \left(\log \sum_{e} e^{\ell_{t,e}}\right)^2 \quad (z\text{-loss})
$$

Biaya aktif per token: $O(D\,d_e \cdot k + D\,d^{\text{sh}})$ — sublinear terhadap $E$,
sehingga menambah expert menambah kapasitas tanpa menambah FLOPs/token.

**Implementasi dispatch.** Dua jalur di `ASCAConfig.moe_impl`:

* `"grouped"` (default) — token diurutkan per expert dan di-pad menjadi blok
  $(E, M, D)$, lalu seluruh expert dijalankan dengan **dua** `bmm` (satu untuk
  $W_1/W_3$ gabungan, satu untuk $W_2$). Kernel launch turun dari $3E$ menjadi 2;
  terukur **1.7×** lebih cepat. Biayanya *padding*: FLOPs routed yang dibayar
  sebesar $E\cdot M / (N k) \ge 1$, bergantung keseimbangan routing
  (terukur 1.07× pada routing seimbang, hingga ~1.65× saat routing miring).
  Nilai ini diekspos sebagai `moe.padding_factor`.
* `"loop"` — gather/scatter per expert; FLOPs persis sebanding token, dipakai
  sebagai referensi kebenaran dan pembanding FLOPs yang adil.

### 3.4 Efficient Memory (SegmentMemory)

Tujuan: konteks panjang dengan anggaran memori terbatas, tanpa KV-cache $O(T)$.

Bagi sekuens menjadi $S = T/L_s$ segmen. Ringkasan segmen $s$ dihitung dengan
$M$ query yang dipelajari $Q^{\text{mem}} \in \mathbb{R}^{M\times D}$:

$$
\mathrm{Sum}^{(s)} = \mathrm{softmax}\!\left(\frac{Q^{\text{mem}} K_s^\top}{\sqrt{D}}\right) V_s \in \mathbb{R}^{M\times D}
$$

Token di segmen $s$ membaca ringkasan **hanya dari segmen $< s$** (kausal antar-segmen):

$$
\mathrm{Mem}^{(s)} = \mathrm{softmax}\!\left(\frac{q_t \,[\mathrm{Sum}^{(0..s-1)}]^\top}{\sqrt{D}}\right)[\mathrm{Sum}^{(0..s-1)}]V
$$

lalu digabung dengan residual bergerbang:

$$
h_t \leftarrow h_t + \sigma(w_m)\odot \mathrm{Mem}^{(s)}_t
$$

Biaya: $O(T M D + S^2 M D)$ — $M \ll T$, jadi jauh di bawah attention penuh.
Kausal: ringkasan segmen $s$ sendiri tidak pernah dibaca oleh token di segmen $s$. ✓

### 3.5 Shared Decoder

$$
\begin{aligned}
z &= \mathrm{RMSNorm}(H^L) \\
z' &= z + \sigma(\gamma)\odot H^{0} \quad (\text{gated semantic skip, } \gamma \text{ init } 0)\\
\mathrm{logits} &= z'\, E^\top \quad (\text{tied embedding})
\end{aligned}
$$

Weight tying memangkas $V\cdot D$ parameter (signifikan pada model kecil) dan
memaksa ruang laten input/output berbagi geometri.

### 3.6 Block

Pre-norm residual, mengikuti pola stabil:

$$
\begin{aligned}
H &\leftarrow H + \mathrm{Attn}(\mathrm{RMSNorm}(H)) \\
H &\leftarrow H + \mathrm{Mem}(\mathrm{RMSNorm}(H)) \\
H &\leftarrow H + \mathrm{MoE}(\mathrm{RMSNorm}(H))
\end{aligned}
$$

---

## 4. Anggaran Komputasi

FLOPs aktif per token (dominasi, $D$ besar):

| Komponen | FLOPs/token |
|---|---|
| Attention lokal | $\approx 4 T_{\text{eff}} D_h H$ dengan $T_{\text{eff}} = \min(T, w)$ |
| Attention linear | $\approx 4 C D_h H + 4 D_h^2 H$ |
| MoE (shared + top-$k$) | $4 D (d^{\text{sh}} + k\, d_e)$ |
| Dense baseline (pembanding) | $4 T D + 8 D d_{ff}$ |

Kunci klaim yang harus diuji: pada **FLOPs/token dan VRAM yang sama**, ASCA harus
mencapai validation loss lebih rendah daripada baseline dense — bukan sekadar
"parameter lebih banyak".

---

## 5. Matriks Ablasi

Setiap baris adalah konfigurasi yang wajib dijalankan pada budget token identik.

| ID | Konfigurasi | Flag |
|---|---|---|
| B0 | Dense Transformer (kontrol) | `baseline/dense_transformer.py` |
| A0 | ASCA penuh | semua `use_*` = true |
| A1 | tanpa Semantic Encoder | `use_semantic_encoder=false` |
| A2 | attention lokal saja | `use_hybrid_attention=false` |
| A3 | tanpa MoE (MLP dense) | `use_moe=false` |
| A4 | tanpa SegmentMemory | `use_memory=false` |
| A5 | MoE tanpa shared expert | `n_shared_experts=0` |
| A6 | tanpa threshold skip | `router_threshold=0.0` |

---

## 6. Protokol Evaluasi

Semua perbandingan pada **jumlah token latih, batch efektif, dan LR schedule identik**.

- **Language**: validation loss & perplexity pada korpus uji yang sama.
- **Efisiensi**: tokens/sec (training), ms/token (decode), active FLOPs/token.
- **Memori**: peak VRAM pada $T \in \{512, 1024, 2048, 4096\}$, dan VRAM decode
  per token (KV-cache growth).
- **Reasoning/coding/math**: GSM8K, HumanEval, MBPP — hanya relevan setelah
  skala cukup; pada tahap prototipe dilaporkan sebagai `n/a`.
- **Statistik**: minimal 3 seed, laporkan mean ± std. Tidak ada klaim dari 1 run.

Hasil ditulis ke `docs/EXPERIMENTS.md` oleh `benchmarks/train_toy.py`
(format JSON + tabel markdown).

---

## 7. Batasan yang Diakui (v0.2)

1. ~~Gating decay per-chunk~~ **selesai (v0.3)**: decay sekarang per token
   ($\lambda_t = \sigma(W_\lambda x_t)$), sehingga bentuk chunked saat training dan
   bentuk rekuren saat decode identik secara matematis.
2. SegmentMemory memakai ringkasan statis per segmen (tidak ada update rekuren
   antar-forward pada versi training paralel).
3. Attention lokal default memakai mask eksplisit (cocok untuk $T \le 4096$); jalur
   FlexAttention tersedia (`local_attn_impl="flex"`) tetapi baru menang saat
   dikompilasi dan pada $T$ besar, jadi belum menjadi default.
4. Belum ada positional encoding pada jalur linear (recency ditangani oleh forget
   gate); ini asumsi desain yang perlu diuji.
5. Grouped GEMM MoE membayar padding yang bergantung keseimbangan routing
   (1.07×–1.65× terukur). Belum ada dispatch dengan kapasitas terbatas per blok.
6. ~~Belum ada KV-cache~~ **selesai (v0.3)** dan ~~latensi decode~~ **selesai (v0.4)**:
   `init_cache`/`forward_step`/`generate_cached` untuk ASCA *dan* baseline, plus
   `asca.graph_decode.GraphDecoder` (satu replay CUDA graph = satu token).
   Decode 8.18 → **1.60 ms/token** (5× lebih cepat), datar terhadap konteks.
   ~~Prefill token-demi-token~~ **selesai (v0.5)**: `ASCAModel.prefill` memproses
   prompt lewat forward chunked lalu menurunkan state cache dari lintasan paralel
   (611 → **48.955 tok/s**, 80×; dense 64.690). Sisa gap: decode 2.5× dan prefill
   1.32× lebih lambat dari dense.
7. `torch.compile` memecah graph pada `int(counts.max().item())` di dispatch MoE
   (graph break, bukan error). Menghapusnya butuh kapasitas blok statis.
8. Belum ada hasil benchmark pada skala bermakna (≥100M token). **Tidak ada klaim
   keunggulan.**
9. Seluruh hasil v0.1–v0.2 memakai `SegmentMemory` dengan normalisasi softmax yang
   salah (lintas posisi query). Sudah diperbaiki di v0.3; angka lama tidak dipakai.
10. ~~Pada teks nyata ASCA kalah dari dense-matched~~ **DICABUT (v0.8)**: angka
   p=0.042 itu memakai satu LR bersama (3e-3). Setelah LR di-tuning per arsitektur,
   selisihnya +0.0098 pada LR optimal masing-masing dan +0.0011 pada lr 1e-2 —
   tidak terbedakan. Yang tetap terukur adalah biayanya: 3.7× lebih lambat dan
   3.4× VRAM lebih besar pada titik kualitas terbaik.
   **Aturan**: LR wajib di-tuning per arsitektur sebelum klaim apa pun.
11. **Ablasi pada korpus sintetis tidak terprediksi.** Di sintetis ASCA tampak
   setara/lebih baik (Δ −0.002…−0.009) dan komponen terberat tampak "beban mati";
   di teks nyata tanda berbalik dan varian ramping jadi +0.09 lebih buruk. Semua
   kesimpulan arsitektural di dokumen ini harus berasal dari korpus nyata.

---

## 8. Catatan Implementasi & Hasil Optimasi Terukur

Semua angka: RTX 4050 Laptop 6 GB, torch 2.11.0+cu128, bf16 autocast.
Detail lengkap + cara reproduksi: `docs/EXPERIMENTS.md`.

| Perubahan | Efek terukur |
|---|---|
| `linear_chunk` 64 → 128 (dengan batas $T/4$) | attention 8.24 → 4.09 ms (2.0×) |
| MoE grouped GEMM (2 `bmm`, bukan 3·E kernel) | MoE 3.78 → 2.22 ms (1.7×) |
| `torch.compile` (opt-in `--compile`) | model penuh 66.3 → 42.9 ms (1.54×) |
| Total end-to-end training (nano, 3 seed) | 63.0k → **106.1k tok/s (1.68×)** |
| KV-cache + state rekuren decode | latensi datar terhadap konteks: 9.3 → 10.3 ms/tok (512 → 2036), cache **3.8 MB konstan** vs dense 24 MB |
| CUDA Graph decode (`asca.graph_decode`) | 8.18 → **1.64 ms/token (5.0×)** |
| Prefill paralel (`ASCAModel.prefill`) | 611 → **48.955 tok/s (80×)**; dense 64.690 (1.32×) |
| Fix normalisasi `SegmentMemory` | val loss ASCA 0.4159 ± 0.0020 vs dense 0.4210 ± 0.0145 (setara) |

### 8.0 Prefill paralel: menurunkan state cache dari lintasan chunked

`prefill` tidak memanggil `forward_step` per token. Setelah forward paralel, state
cache diturunkan dari lintasan yang sama:

| Komponen | State yang diturunkan |
|---|---|
| attention lokal | ring buffer: posisi `T-w..T-1` ditulis ke slot `pos % w` |
| attention linear | `S`, `ksum` akhir dari loop chunk (dikembalikan `_global_impl`) |
| memori | ringkasan `T // L_s` segmen selesai + akumulator online-softmax segmen aktif |
| encoder | buffer `kernel` input ternormalisasi terakhir |

Konsistensi diuji: `prefill` lalu decode == loop `forward_step` T kali (galat ~1e-7).

### 8.1 Syarat agar jalur decode bisa di-CUDA-graph

Supaya satu graph bisa dipakai ulang untuk semua posisi, jalur `step` harus
**bebas cabang data-dependent**:

1. Posisi/window memakai `scatter_`/`index_select` dengan indeks dari **tensor**
   (`pos % w`), bukan `if`/irisan Python.
2. Ringkasan segmen dihitung inkremental dengan online softmax
   (`acc`, `l`, `m` berjalan) — bukan dihitung ulang tiap token.
3. MoE memakai dispatch terpilih (`index_select` atas bobot expert), tanpa
   `nonzero`/`.item()` yang membuat bentuk tensor bergantung data.
4. **Semua update state in-place.** Kalau buffer di-rebind ke tensor baru saat
   capture, memori lama dilepas di tengah capture dan graph bisa menulis ke blok
   yang masih dipakai tensor lain (aliasing → `device-side assert` saat replay).
5. Tidak ada `.item()`/sinkronisasi CPU–GPU di dalam langkah; penghitung posisi
   disimpan sebagai tensor dan dinaikkan in-place.

`torch.compile` **bukan** solusi untuk jalur ini: terukur 40× lebih lambat
(guards + recompiles) dibanding CUDA graph.

Status jujur: kualitas setara dense param-matched (0.4165 vs 0.4167 val loss pada
1.23M token), FLOPs/token +7% pada active params setara (preset tiny), tetapi
**wall-clock masih ~2.6–5× lebih lambat** dan **VRAM ~1.9× lebih besar**. Penyebab
utamanya adalah occupancy kernel pada `d_model` kecil, bukan FLOPs.

