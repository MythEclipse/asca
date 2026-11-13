# ASCA v2 — spesifikasi arsitektur (tanpa attention, state konstan)

Status: **implementasi Rust jalan** (training + loading + sampling), belum ada klaim
keunggulan apa pun terhadap baseline.

## 1. Mengapa v2 ada

v1 adalah varian Transformer yang ditulis di atas PyTorch. Tiga masalah mendasar:

1. **Klaim efisiensinya tidak sah.** Yang terukur adalah PyTorch, bukan arsitekturnya.
   Decode eager butuh ~2.900 kernel launch per token; gap training 3.7× diukur pada
   `d=128` tempat overhead dispatcher dan GEMM kecil mendominasi.
2. **Tidak cukup berbeda.** v1 tetap memakai attention, softmax, tokenizer, dan router
   MoE terlatih. Even kalau menang, itu menang tipis di dalam keluarga yang sama.
3. **Tidak bisa dipakai.** Semua hasil berhenti di skrip riset Python.

v2 dibangun dari nol di Rust: training, loading, dan inferensi. Tanpa dependensi
eksternal. Setiap gradien diturunkan tangan dan diverifikasi.

## 2. Desain

| Aspek | v1 (PyTorch) | **v2 (Rust murni)** |
|---|---|---|
| Attention | hybrid softmax + linear | **tidak ada** |
| Softmax | ada | **nol di jalur fitur** |
| State | KV-cache tumbuh O(T) | **S ∈ R^{d×d} per head, konstan** |
| Tokenizer | byte via library | **byte-level, tanpa tokenizer** |
| Router MoE | terlatih, top-k, aux loss | **hash(id) % n_experts**, nol FLOP |
| Training | autograd PyTorch | **backward tulisan tangan (BPTT)** |
| Loading | `torch.load` | **format biner sendiri** |
| Dependensi | torch, numpy, tokenizers | **nol** |

### 2.1 Inti: gated delta rule

Setiap head menyimpan memori asosiatif `S ∈ R^{d_h × d_h}`:

```text
Sk_t = S_{t-1}^T k_t
u_t  = beta_t * (v_t - alpha_t * Sk_t)
S_t  = alpha_t * S_{t-1} + k_t u_t^T
o_t  = S_t^T q_t
```

Suku `(v_t - alpha_t S^T k_t)` adalah pembeda dari linear attention biasa: state
**menulis ulang** slot yang sudah ada (overwrite), bukan menumpuk buta. `alpha_t`
forget gate, `beta_t` laju tulis.

Memori decode: `L * H * d_h^2` float, **tidak bergantung panjang konteks**.

### 2.2 Bentuk chunked (training) dan paritasnya

Training memakai bentuk paralel per chunk dengan transformasi UT. Dengan
`gamma_t = prod_{i<=t} alpha_i` dan `rho_t = gamma_t / gamma_C` (decay relatif,
supaya `1/rho <= 1` dan tidak meledak):

```text
L[j,i] = beta_j * rho_j * (k_j . k_i / rho_i)        untuk i < j
U      = (I + L)^-1 diag(beta) (V - gamma_C * rho * (K S_0))
O      = rho * (gamma_C * (Q S_0) + tril(Q Ktilde^T, 0) U)
S_end  = gamma_C * S_0 + Ktilde^T U
```

Bentuk rekuren dan chunked **identik sampai 1.8e-15** (fp64, chunk 8/16/40/64) —
diverifikasi di `tests/test_delta.py`. Ini gerbang wajib: bug paritas sudah dua kali
menggigit proyek di v1.

Catatan implementasi: `1/gamma` mentah bisa sekecil 1e-40 pada chunk panjang dan
membuat hasil meledak (galat 1e+05). Decay relatif `rho` adalah perbaikan yang
membuatnya stabil.

### 2.3 Hash-routed MoE

Expert dipilih `hash(id_token) % n_experts`. Konsekuensi:

- **nol FLOP routing** (tidak ada GEMM router)
- **tidak ada aux load-balancing loss** (beban seimbang by construction)
- **tidak ada capacity factor / token drop**
- trade-off: tidak ada pembelajaran routing, kualitas bergantung pada hash yang baik

### 2.4 Blok

```text
x = x + DeltaBlock(RMSNorm(x))
x = x + HashMoE(RMSNorm(x))
```

Kepala akhir: `(d_model -> 256)`, loss = cross-entropy next-byte.

## 3. Implementasi Rust

```text
rust/
  Cargo.toml
  src/tensor.rs   Mat, RNG xorshift128+, RMSNorm fwd/bwd, AdamW (decoupled wd)
  src/delta.rs    delta block: forward + backward BPTT tulisan tangan
  src/model.rs    embed -> N x [delta, hash-MoE] -> RMSNorm -> head, CE + backward
  src/train.rs    data byte-level, train loop, eval, checkpoint biner
  src/main.rs     CLI: train / eval / sample / inspect
  tests/gradcheck.rs        beda hingga pada delta block
  tests/gradcheck_model.rs  beda hingga pada model penuh
```

Tanpa dependensi eksternal. `cargo build --release` cukup.

### 3.1 Verifikasi gradien

Backward diturunkan manual, jadi diverifikasi dengan beda hingga pada entri
**bergradien terbesar** (bukan sampel acak), dengan kriteria sadar-derau:

- `truncation ~ eps^2 * f'''` turun kalau `eps` kecil
- `derau f32 ~ ulp(loss) / (2 eps)` naik kalau `eps` kecil

`eps = 3e-4` dipilih di titik seimbang. Entri di bawah 3× derau dilewati, bukan
divonis gagal. Hasil: 40 entri di semua parameter model cocok, galat relatif
terburuk 0.26 dari toleransi.

Dua bug nyata ditemukan lewat proses ini:
1. `dx` punya suku palsu `d_o @ W_o^T` — `o` bukan fungsi langsung dari `x`.
2. Backward L2-normalize butuh norm pra-normalisasi yang disimpan; tanpa itu
   gradien q/k salah.

### 3.2 Checkpoint

Format biner: magic `ASCA2CK1`, 6 angka konfigurasi (u64 LE), lalu tiap parameter
(`u64` jumlah elemen + f32 LE). `asca inspect` membacanya kembali tanpa model asli.

## 4. Hasil terukur sejauh ini

TinyStories byte-level, CPU (tanpa GPU), `d_model=64 n_heads=4 n_layers=2
n_experts=8 d_ff=128`:

| Metrik | Nilai |
|---|---|
| Parameter | 331.408 |
| Throughput training | **6.583 tok/s** (CPU, matmul naif) |
| Token dilatih | 614k (1200 langkah, batch 4, seq 128) |
| val_loss | 2.96 → **1.379** (ppl 3.97) |
| Memori decode | state konstan `L*H*d_h^2` = 2×4×16² = 2048 float |

## 5. Yang BELUM dibuktikan (jujur)

1. **Tidak ada perbandingan dengan baseline di engine yang sama.** Ini langkah
   berikutnya yang paling penting: baseline dense harus diimplementasikan di crate
   yang sama, dengan kernel yang sama, supaya perbandingan tidak bias framework.
   Tanpa itu, tidak ada klaim efisiensi yang sah — persis kesalahan v1.
2. **Belum ada kernel CUDA.** 6.583 tok/s itu CPU dengan matmul naif. Kernel
   (int8/f16, tensor core, fused elementwise) belum ditulis.
3. **Belum ada kurva loss vs token** — belum tahu apakah delta rule menyusul dense
   pada skala lebih besar.
4. **Gate `w_gate`/`b_gate` menerima gradien sangat kecil** (skala 4.6e-4 vs 1.3e0
   untuk `b_qkv`). Perlu diperiksa: apakah gate belajar, atau perlu inisialisasi/lr
   terpisah.
5. **Sample masih gibberish** setelah 614k token — wajar untuk skala ini, tapi belum
   diukur terhadap baseline.

## 6. Aturan yang diwarisi dari v1

- LR di-tuning **per arsitektur** sebelum klaim apa pun.
- Ablasi korpus sintetis **tidak** dipakai untuk keputusan arsitektur.
- Klaim butuh mean ± std, ≥3 seed, dan noise floor terukur.
- Tidak ada klaim keunggulan sebelum baseline di engine yang sama dijalankan.
