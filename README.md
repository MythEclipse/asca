# ASCA — Adaptive Sparse Cognitive Architecture

Proyek penelitian dan pengembangan arsitektur neural network LLM eksperimental dari nol (scratch), dirancang untuk mencapai efisiensi komputasi, pemahaman semantik mendalam, dan footprint VRAM rendah melalui mekanisme adaptif dan sparse.

---

## 1. Visi Utama

Membangun foundation model general-purpose yang berfokus pada:
- **Language & Semantic Primacy**: Pemahaman makna dan relasi bahasa sebagai prioritas utama.
- **Reasoning & Logic**: Kemampuan berpikir deduktif dan logis.
- **Algorithms & Code**: Pemahaman struktur logika pemrograman.
- **Mathematics & Science**: Kemampuan komputasi simbolik dan saintifik.
- **Long-Context Efficiency**: Skalabilitas konteks panjang tanpa ledakan memori O(N^2).
- **High-Throughput Inference**: Latensi decode rendah dengan konsumsi VRAM minimal pada GPU skala konsumen.

---

## 2. Prinsip Pengembangan

1. **Dari Nol (Tabula Rasa)**: Semua bobot diinisialisasi secara acak (*random initialization*); tidak menggunakan checkpoint/bobot pretrained.
2. **Mekanisme Mandiri**: Merumuskan arsitektur dan operator internal sendiri, bukan sekadar memodifikasi konfigurasi Transformer standar.
3. **Efisiensi sebagai Metrik Utama**: Mengukur kualitas bukan dari jumlah parameter mentah, melainkan dari rasio performa terhadap FLOPs dan VRAM.
4. **Validasi Berbasis Kontrol**: Setiap klaim keunggulan wajib diuji secara head-to-head terhadap baseline kontrol (Dense Transformer dan MoE standar) pada budget FLOPs/token yang sama.

---

## 3. Komponen Inti ASCA

| Komponen | Peran Fungsional |
|---|---|
| **Semantic Encoder** | Memahami makna dan relasi representasi bahasa awal secara kausal |
| **Adaptive Router** | Memilih jalur komputasi secara dinamis sesuai kompleksitas token/task |
| **Sparse Experts** | Memproses representasi tugas secara selektif (terinspirasi MoE) |
| **Efficient Memory** | Mempertahankan dependensi konteks panjang dengan footprint VRAM terkontrol |
| **Adaptive Attention** | Memangkas komputasi attention yang redundan secara terarah |
| **Shared Decoder** | Merekonstruksi representasi laten menjadi distribusi token probabilitas |

---

## 4. Struktur Repositori

```
asca/
├── docs/                      # Spesifikasi matematis, RFC, dan log eksperimen
│   ├── SPEC_ASCA.md           # Notasi matematis formal dan dataflow tensor
│   └── EXPERIMENTS.md         # Hasil pengujian empiris vs baseline
├── asca/                      # Implementasi modul arsitektur ASCA
│   ├── __init__.py
│   ├── encoder.py
│   ├── router.py
│   ├── experts.py
│   ├── attention.py
│   ├── memory.py
│   └── model.py
├── baseline/                  # Model kontrol pembanding (Dense Causal Transformer)
│   ├── __init__.py
│   └── dense_transformer.py
├── data/                      # Tokenizer dan streaming dataloader
│   └── __init__.py
└── benchmarks/                # Pengukuran FLOPs, VRAM, latency, dan toy training
    └── __init__.py
```

---

## 5. Protokol Evaluasi

Semua perbandingan performa diukur menggunakan:
- **Language**: Perplexity & Validation Loss pada korpus uji yang sama.
- **Reasoning & Code**: Akurasi penalaran logis, GSM8K, dan benchmark coding (HumanEval/MBPP).
- **Compute Efficiency**: Active FLOPs per token, throughput (tokens/sec), dan latensi decode (ms/tok).
- **Memory Footprint**: Peak allocated VRAM pada berbagai sequence lengths (512, 1024, 2048, 4096).
