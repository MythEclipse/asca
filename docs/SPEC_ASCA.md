# ASCA Specification: Mathematical Formulation & Tensor Flow

Versi: 0.1-draft
Status: Desain Awal

Dokumen ini mendefinisikan formalisasi matematis untuk komponen Adaptive Sparse Cognitive Architecture (ASCA).

---

## 1. Notasi Dasar

- Dimensi Batch: $B$
- Panjang Sekuens: $T$
- Dimensi Model (Hidden Dim): $D$
- Jumlah Expert: $E$, dengan top-$k$ routing: $k \ll E$
- Vocabulary Size: $V$

---

## 2. Peta Alur Tensor (Causal Dataflow)

Input Token: $X \in \mathbb{N}^{B 	imes T}$
Embedding: $H_0 = 	ext{Embed}(X) \in \mathbb{R}^{B 	imes T 	imes D}$

$$\dots 	ext{To be detailed} \dots$$
