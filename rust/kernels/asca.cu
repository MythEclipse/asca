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
// Satu blok per (head). State dh x dh di shared memory; loop waktu di dalam kernel.
// Forward: q,k sudah dinormalisasi L2 per head, alpha/beta per (token, head).
extern "C" __global__ void delta_fwd_f32(const float* __restrict__ q,  // (T, H*dh)
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         const float* __restrict__ alpha,  // (T, H)
                                         const float* __restrict__ beta,
                                         float* __restrict__ out,  // (T, H*dh)
                                         int T, int H, int dh) {
    const int h = blockIdx.x;
    const int off = h * dh;
    extern __shared__ float sh[];  // dh*dh state + 3*dh temp
    float* S = sh;
    float* sk = sh + dh * dh;
    float* u = sk + dh;
    float* kt = u + dh;
    for (int i = threadIdx.x; i < dh * dh; i += blockDim.x) S[i] = 0.0f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const float* qt = q + (long)t * H * dh + off;
        const float* ktt = k + (long)t * H * dh + off;
        const float* vt = v + (long)t * H * dh + off;
        const float a = alpha[(long)t * H + h];
        const float b = beta[(long)t * H + h];
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
        float* ot = out + (long)t * H * dh + off;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            float acc = 0.0f;
            for (int a2 = 0; a2 < dh; ++a2) acc += S[a2 * dh + i] * qt[a2];
            ot[i] = acc;
        }
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
