//! Kernel CUDA untuk ASCA v2.
//!
//! Aturan desain:
//!  * tidak ada softmax dan tidak ada attention di arsitektur ini, jadi kernel yang
//!    dibutuhkan hanya GEMM, elementwise, dan satu scan sekuensial untuk delta rule;
//!  * delta rule dipetakan satu blok per head: state `dh x dh` muat di shared memory
//!    sehingga seluruh loop waktu berjalan di dalam satu kernel tanpa keluaran antara;
//!  * semua kernel f32 dulu; jalur tensor core f16 menyusul setelah paritasnya terbukti.

#define TS 32

// ---------------------------------------------------------------- GEMM dasar
// C[M,N] = A[M,K] @ B[K,N], tile 32x32 di shared memory.
extern "C" __global__ void matmul_f32(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TS][TS + 1];
    __shared__ float Bs[TS][TS + 1];
    const int row = blockIdx.y * TS + threadIdx.y;
    const int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        const int ka = k0 + threadIdx.x;
        const int kb = k0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ka < K) ? A[(long)row * K + ka] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (kb < K && col < N) ? B[(long)kb * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(long)row * N + col] = acc;
}

// C[M,N] = A[K,M]^T @ B[K,N] — dipakai untuk dW = x^T @ dy.
extern "C" __global__ void matmul_ta_f32(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TS][TS + 1];
    __shared__ float Bs[TS][TS + 1];
    const int row = blockIdx.y * TS + threadIdx.y;
    const int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        const int ka = k0 + threadIdx.y;
        const int kb = k0 + threadIdx.y;
        // As[i][j] = A[(k0+j) * M + row]  (transpose saat memuat)
        As[threadIdx.y][threadIdx.x] =
            (row < M && (k0 + threadIdx.x) < K) ? A[(long)(k0 + threadIdx.x) * M + row] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (kb < K && col < N) ? B[(long)kb * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(long)row * N + col] = acc;
}

// GEMM dengan leading dimension + offset kolom pada B:
//   C[M,N] = A[M,K] @ B[:, offb : offb+N],  B berpindah baris sejauh `ldb`.
// Dipakai MoE: satu blok bobot `w1` (d x n_experts*d_ff) dipakai semua expert, tiap
// expert hanya menggeser `offb`. Tanpa ini tiap expert butuh salinan kolom sendiri di
// host (d x d_ff) yang lalu diunggah ulang — 8 salinan per layer.
extern "C" __global__ void matmul_ld_f32(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C, int M, int K, int N, int ldb,
                                         int offb) {
    __shared__ float As[TS][TS + 1];
    __shared__ float Bs[TS][TS + 1];
    const int row = blockIdx.y * TS + threadIdx.y;
    const int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        const int ka = k0 + threadIdx.x;
        const int kb = k0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? A[(long)row * K + ka] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (kb < K && col < N) ? B[(long)kb * ldb + offb + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(long)row * N + col] = acc;
}

// ------------------------------------------------------------------ elementwise
extern "C" __global__ void add_bias_f32(float* __restrict__ X, const float* __restrict__ b,
                                        int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) X[i] += b[i % cols];
}

extern "C" __global__ void axpy_f32(float* __restrict__ Y, const float* __restrict__ X,
                                    float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] += a * X[i];
}

extern "C" __global__ void scale_f32(float* __restrict__ X, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] *= a;
}

// RMSNorm per baris; `inv` menyimpan 1/rms untuk dipakai backward.
extern "C" __global__ void rmsnorm_f32(const float* __restrict__ X, float* __restrict__ Y,
                                       float* __restrict__ inv, int rows, int cols,
                                       float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = X + (long)r * cols;
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.0f;
    __syncthreads();
    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local += row[c] * row[c];
    atomicAdd(&ssum, local);
    __syncthreads();
    const float iv = rsqrtf(ssum / (float)cols + eps);
    if (threadIdx.x == 0) inv[r] = iv;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) Y[(long)r * cols + c] = row[c] * iv;
}

// Backward RMSNorm: dx = inv * (dy - xhat * mean(dy * xhat)).
extern "C" __global__ void rmsnorm_bwd_f32(const float* __restrict__ X,
                                           const float* __restrict__ DY,
                                           float* __restrict__ DX, int rows, int cols,
                                           float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = X + (long)r * cols;
    const float* dyr = DY + (long)r * cols;
    __shared__ float ssum;
    __shared__ float smean;
    if (threadIdx.x == 0) { ssum = 0.0f; smean = 0.0f; }
    __syncthreads();
    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local += row[c] * row[c];
    atomicAdd(&ssum, local);
    __syncthreads();
    const float iv = rsqrtf(ssum / (float)cols + eps);
    float lm = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) lm += dyr[c] * row[c] * iv;
    atomicAdd(&smean, lm);
    __syncthreads();
    const float mean = smean / (float)cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        DX[(long)r * cols + c] = iv * (dyr[c] - row[c] * iv * mean);
}

// ------------------------------------------------------- delta rule (scan waktu)
// Satu blok per (head, sekuens). State dh x dh di shared memory, loop waktu di dalam
// kernel.
//
// Baris masukan ditumpuk `B * seq` (batching lintas sekuens), jadi tiap blok hanya
// membaca baris `bi * seq + t` miliknya sendiri dan me-reset state di awal: tidak ada
// token yang bocor antar sekuens. Versi sebelumnya memakai grid satu dimensi per head
// dan menelusuri seluruh T baris sebagai satu sekuens — benar hanya kalau B = 1, dan
// diam-diam salah (tanpa error) begitu batching dipakai.
//
// `states` (boleh nullptr) menyimpan state SEBELUM tiap langkah plus state akhir,
// layout `((h * B) + bi) * (seq + 1) * dh * dh`, persis yang dibutuhkan BPTT.
extern "C" __global__ void delta_fwd_batch_f32(const float* __restrict__ q,  // (B*seq, H*dh)
                                               const float* __restrict__ k,
                                               const float* __restrict__ v,
                                               const float* __restrict__ alpha,  // (B*seq, H)
                                               const float* __restrict__ beta,
                                               float* __restrict__ out,  // (B*seq, H*dh)
                                               float* __restrict__ states,
                                               int H, int dh, int B, int seq) {
    const int h = blockIdx.x;
    const int bi = blockIdx.y;
    const int off = h * dh;
    const int row0 = bi * seq;
    extern __shared__ float sh[];  // dh*dh state + 3*dh temp
    float* S = sh;
    float* sk = sh + dh * dh;
    float* u = sk + dh;
    float* kt = u + dh;
    for (int i = threadIdx.x; i < dh * dh; i += blockDim.x) S[i] = 0.0f;
    __syncthreads();
    float* st = states ? states + ((long)(h * B) + bi) * (long)(seq + 1) * dh * dh : nullptr;
    for (int t = 0; t < seq; ++t) {
        const long r = (long)row0 + t;
        if (st) {
            float* sdst = st + (long)t * dh * dh;
            for (int i = threadIdx.x; i < dh * dh; i += blockDim.x) sdst[i] = S[i];
        }
        const float* qt = q + r * H * dh + off;
        const float* ktt = k + r * H * dh + off;
        const float* vt = v + r * H * dh + off;
        const float a = alpha[r * H + h];
        const float b = beta[r * H + h];
        for (int i = threadIdx.x; i < dh; i += blockDim.x) kt[i] = ktt[i];
        __syncthreads();
        // sk[i] = sum_a S[a][i] * k[a]
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            float acc = 0.0f;
            for (int a2 = 0; a2 < dh; ++a2) acc += S[a2 * dh + i] * kt[a2];
            sk[i] = acc;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < dh; i += blockDim.x) u[i] = b * (vt[i] - a * sk[i]);
        __syncthreads();
        // S = a*S + k u^T ; o = S^T q
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) {
            const int a2 = idx / dh;
            const int i = idx % dh;
            S[idx] = a * S[idx] + kt[a2] * u[i];
        }
        __syncthreads();
        float* ot = out + r * H * dh + off;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            float acc = 0.0f;
            for (int a2 = 0; a2 < dh; ++a2) acc += S[a2 * dh + i] * qt[a2];
            ot[i] = acc;
        }
        __syncthreads();
    }
    if (st) {
        float* slast = st + (long)seq * dh * dh;
        for (int i = threadIdx.x; i < dh * dh; i += blockDim.x) slast[i] = S[i];
    }
}

// ------------------------------------------------- delta rule: backward (BPTT)
// Satu blok per (head, sekuens), memakai state yang ditulis `delta_fwd_batch_f32` dan
// TIDAK pernah menyeberang ke host. Ini bagian yang membuat offload scan masuk akal:
// kalau state harus diunduh dulu supaya BPTT bisa jalan di CPU, biaya salinannya
// (ratusan MB per langkah) lebih besar daripada scan CPU-nya sendiri.
//
// Turunannya sama persis dengan versi Rust di `src/delta.rs` (yang diverifikasi
// gradcheck), termasuk urutan operasinya:
//   dq_a   = dot(do_t, baris_a(S_t))
//   ds    += q_a * do_t
//   sk     = S_prev^T k          (dipakai untuk u)
//   u      = beta * (v - alpha*sk)
//   dalpha = <ds, S_prev> - beta*<du, sk>
//   dbeta  = <du, v - alpha*sk>
//   dk_a  += <ds_a, u> + <S_prev_a, dsk>
//   du     = sum_a k_a * ds_a
//   dsk    = -beta*alpha*du
//   ds_prev = alpha*ds + k dsk^T
//
// Keluaran `d_qkv` (T, 3*d) sudah termasuk backward L2-normalisasi per head, dan
// `d_gate` (T, 2H) sudah termasuk backward sigmoid — jadi tidak ada kernel tambahan.
extern "C" __global__ void delta_bwd_f32(const float* __restrict__ states,
                                         const float* __restrict__ q,  // (T, H*dh) mentah
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         const float* __restrict__ alpha,  // (T, H)
                                         const float* __restrict__ beta,
                                         const float* __restrict__ qnorm,  // (T, H)
                                         const float* __restrict__ knorm,
                                         const float* __restrict__ d_o,  // (T, H*dh)
                                         float* __restrict__ d_qkv,      // (T, 3*H*dh)
                                         float* __restrict__ d_gate,     // (T, 2*H)
                                         int H, int dh, int B, int seq) {
    const int h = blockIdx.x;
    const int bi = blockIdx.y;
    const int off = h * dh;
    const int d = H * dh;
    const int row0 = bi * seq;
    extern __shared__ float sh[];
    float* ds = sh;                  // dh*dh
    float* dsprev = ds + dh * dh;    // dh*dh
    float* sk = dsprev + dh * dh;    // dh
    float* u = sk + dh;              // dh
    float* dqv = u + dh;             // dh  (dq sementara)
    float* dk = dqv + dh;            // dh
    float* du = dk + dh;             // dh
    float* dsk = du + dh;            // dh
    float* qs = dsk + dh;            // dh  q ternormalisasi
    float* ks = qs + dh;             // dh  k ternormalisasi
    float* red = ks + dh;            // blockDim.x (reduksi)

    const float* st = states + ((long)(h * B) + bi) * (long)(seq + 1) * dh * dh;
    for (int i = threadIdx.x; i < dh * dh; i += blockDim.x) ds[i] = 0.0f;
    __syncthreads();
    for (int t = seq - 1; t >= 0; --t) {
        const long r = (long)row0 + t;
        const float* s_t = st + (long)(t + 1) * dh * dh;
        const float* s_prev = st + (long)t * dh * dh;
        const float* qt_raw = q + r * H * dh + off;
        const float* kt_raw = k + r * H * dh + off;
        const float* vt = v + r * H * dh + off;
        const float* do_t = d_o + r * H * dh + off;
        const float a = alpha[r * H + h];
        const float b = beta[r * H + h];
        // q dan k yang dikirim ke kernel SUDAH dinormalisasi L2 per head oleh
        // `DeltaBlock::forward` (normalisasi terjadi in-place sebelum scan). Jadi di sini
        // hanya disalin; `qn_inv`/`kn_inv` dipakai nanti untuk backward normalisasi.
        const float qn_inv = 1.0f / fmaxf(qnorm[r * H + h], 1e-8f);
        const float kn_inv = 1.0f / fmaxf(knorm[r * H + h], 1e-8f);
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            qs[i] = qt_raw[i];
            ks[i] = kt_raw[i];
            dk[i] = 0.0f;  // dk WAJIB direset tiap timestep
        }
        if (threadIdx.x < 2) red[threadIdx.x] = 0.0f;
        __syncthreads();

        // dq_a = dot(do_t, baris_a(S_t))
        if (threadIdx.x < dh) {
            const int a2 = threadIdx.x;
            float acc = 0.0f;
            for (int i = 0; i < dh; ++i) acc += do_t[i] * s_t[a2 * dh + i];
            dqv[a2] = acc;
        }
        __syncthreads();
        // ds += q do_t^T
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) {
            ds[idx] += qs[idx / dh] * do_t[idx % dh];
        }
        __syncthreads();
        // sk_i = sum_a k_a S_prev[a][i]
        if (threadIdx.x < dh) {
            const int i = threadIdx.x;
            float acc = 0.0f;
            for (int a2 = 0; a2 < dh; ++a2) acc += ks[a2] * s_prev[a2 * dh + i];
            sk[i] = acc;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < dh; i += blockDim.x) u[i] = b * (vt[i] - a * sk[i]);
        __syncthreads();
        // dalpha = <ds, S_prev> ; dk_a += <ds_a, u> ; du_i = sum_a k_a ds[a][i]
        float dalpha_l = 0.0f;
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) {
            dalpha_l += ds[idx] * s_prev[idx];
        }
        if (threadIdx.x < dh) {
            const int a2 = threadIdx.x;
            const int i = threadIdx.x;
            float acc_u = 0.0f;
            float acc_du = 0.0f;
            for (int j = 0; j < dh; ++j) {
                acc_u += ds[a2 * dh + j] * u[j];
                acc_du += ks[j] * ds[j * dh + i];
            }
            dk[a2] += acc_u;
            du[i] = acc_du;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) dsprev[idx] = a * ds[idx];
        // dbeta = <du, v - alpha*sk> ; dalpha -= beta*<du, sk> ; dv = beta*du
        float dbeta_l = 0.0f;
        float dalpha2_l = 0.0f;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            dbeta_l += du[i] * (vt[i] - a * sk[i]);
            dalpha2_l += -b * du[i] * sk[i];
            d_qkv[r * 3 * d + 2 * d + off + i] = b * du[i];
            dsk[i] = -b * a * du[i];
        }
        __syncthreads();
        atomicAdd(&red[0], dalpha_l + dalpha2_l);
        atomicAdd(&red[1], dbeta_l);
        // dk_a += <S_prev_a, dsk> ; ds_prev += k dsk^T
        if (threadIdx.x < dh) {
            const int a2 = threadIdx.x;
            float acc = 0.0f;
            for (int i = 0; i < dh; ++i) acc += s_prev[a2 * dh + i] * dsk[i];
            dk[a2] += acc;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) {
            dsprev[idx] += ks[idx / dh] * dsk[idx % dh];
        }
        __syncthreads();
        // d_gate: turunan sigmoid
        if (threadIdx.x == 0) {
            d_gate[r * 2 * H + h] = red[0] * a * (1.0f - a);
            d_gate[r * 2 * H + H + h] = red[1] * b * (1.0f - b);
        }
        // backward L2-normalisasi: d_raw = (d_norm - q_hat <q_hat, d_norm>) / norma
        // (norma di sini adalah norma sebelum normalisasi; q_hat = q_raw / norma)
        if (threadIdx.x == 0) {
            float dq_dot = 0.0f;
            float dk_dot = 0.0f;
            for (int i = 0; i < dh; ++i) {
                dq_dot += qs[i] * dqv[i];
                dk_dot += ks[i] * dk[i];
            }
            for (int i = 0; i < dh; ++i) {
                d_qkv[r * 3 * d + off + i] = (dqv[i] - qs[i] * dq_dot) * qn_inv;
                d_qkv[r * 3 * d + d + off + i] = (dk[i] - ks[i] * dk_dot) * kn_inv;
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < dh * dh; idx += blockDim.x) ds[idx] = dsprev[idx];
        __syncthreads();
    }
}

// ------------------------------------------------ cross-entropy + softmax backward
// 1 blok per baris (token t), 256 threads per block.
// Menghitung loss per token dan mengisi dlogits[t, :] = (p - 1_target) * inv_T.
extern "C" __global__ void cross_entropy_bwd_f32(const float* __restrict__ logits,
                                                const unsigned int* __restrict__ targets,
                                                float* __restrict__ dlogits,
                                                float* __restrict__ losses,
                                                int T, int V, float inv_T) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* row = logits + (long)t * V;
    float* drow = dlogits + (long)t * V;
    const unsigned int target = targets[t];

    __shared__ float sbuf[256];

    // 1. cari max lokal
    float lmax = -1e38f;
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        float val = row[c];
        if (val > lmax) lmax = val;
    }
    sbuf[threadIdx.x] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sbuf[threadIdx.x + s] > sbuf[threadIdx.x]) {
                sbuf[threadIdx.x] = sbuf[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    const float max_val = sbuf[0];
    __syncthreads();

    // 2. sum exp
    float lsum = 0.0f;
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        lsum += expf(row[c] - max_val);
    }
    sbuf[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float sum_exp = sbuf[0];
    const float lse = max_val + logf(sum_exp);

    if (threadIdx.x == 0) {
        losses[t] = (target < (unsigned int)V) ? (lse - row[target]) : 0.0f;
    }

    // 3. dlogits = (exp(row - lse) - (c == target ? 1 : 0)) * inv_T
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        float p = expf(row[c] - lse);
        if (c == target) p -= 1.0f;
        drow[c] = p * inv_T;
    }
}

// ------------------------------------------------------------------ AdamW
extern "C" __global__ void adamw_f32(float* __restrict__ val,
                                     const float* __restrict__ grad,
                                     float* __restrict__ m,
                                     float* __restrict__ v,
                                     float lr, float beta1, float beta2,
                                     float eps_t, float wd, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = grad[idx];
    float p = val[idx];
    if (wd > 0.0f) {
        p -= lr * wd * p;
    }
    float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_val;
    v[idx] = v_val;
    p -= lr * m_val / (sqrtf(v_val) + eps_t);
    val[idx] = p;
}

// ------------------------------------------------------- gather / scatter / silu
// MoE mengelompokkan token per expert. Supaya expert bisa dihitung sebagai GEMM
// biasa (bukan loop per baris), baris token dikumpulkan dulu ke buffer kontigu,
// dihitung, lalu disebar kembali ke posisi aslinya.
extern "C" __global__ void gather_rows_f32(const float* __restrict__ src,
                                           const int* __restrict__ idx,
                                           float* __restrict__ dst, int n_rows, int cols) {
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const float* s = src + (long)idx[r] * cols;
    float* d = dst + (long)r * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) d[c] = s[c];
}

extern "C" __global__ void scatter_rows_f32(const float* __restrict__ src,
                                            const int* __restrict__ idx,
                                            float* __restrict__ dst, int n_rows, int cols) {
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const float* s = src + (long)r * cols;
    float* d = dst + (long)idx[r] * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) d[c] = s[c];
}

extern "C" __global__ void silu_f32(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}
