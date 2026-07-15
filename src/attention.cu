// FlashAttention:  O = softmax(Q Kᵀ / √d) V   —  WITHOUT ever building the n×n
// score matrix. That matrix is the whole problem: at seqlen 128k it would be tens
// of TB. FlashAttention streams over the keys keeping a *running* max and sum (an
// online softmax) and accumulates O incrementally, so memory is O(d), not O(n²).
//
// The online-softmax core here is exactly the reduction from the softmax worklog's
// v6 kernel — here it's wrapped around the two matmuls (Q·Kᵀ and P·V).
//
//   for each key j:
//       s      = (q · k_j) * scale
//       m_new  = max(m, s)                 # running max
//       corr   = exp(m - m_new)            # rescale what we had
//       p      = exp(s - m_new)
//       l      = l*corr + p                # running denominator
//       acc    = acc*corr + p * v_j        # running numerator (the output)
//       m      = m_new
//   O = acc / l
//
// v1: one thread per query row, streaming over keys. Correct and O(d) memory —
// the real FlashAttention idea — just not yet tiled into shared memory (that's v2).
// Layout: Q,K,V,O are [n_heads, seqlen, head_dim], one batch. Causal.
//
// Build & run:  nvcc -O3 -arch=sm_90 src/attention.cu -o attention && ./attention

#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define HEAD_DIM 64          // TinyLlama's head_dim; acc fits in registers

__global__ void flash_attn_v1(const float* Q, const float* K, const float* V,
                              float* O, int seqlen, float scale) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;   // which query row
    int head = blockIdx.y;
    if (qi >= seqlen) return;

    const float* q = Q + ((size_t)head * seqlen + qi) * HEAD_DIM;
    const float* Kh = K + (size_t)head * seqlen * HEAD_DIM;
    const float* Vh = V + (size_t)head * seqlen * HEAD_DIM;

    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;
    float m = -FLT_MAX, l = 0.0f;

    // causal: query qi attends to keys 0..qi
    for (int kj = 0; kj <= qi; kj++) {
        const float* k = Kh + (size_t)kj * HEAD_DIM;
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) s += q[d] * k[d];
        s *= scale;

        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * corr + p;
        const float* v = Vh + (size_t)kj * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) acc[d] = acc[d] * corr + p * v[d];
        m = m_new;
    }

    float* o = O + ((size_t)head * seqlen + qi) * HEAD_DIM;
    float inv_l = 1.0f / l;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o[d] = acc[d] * inv_l;
}

// v2: tile queries AND keys. A block owns Br query rows (one thread each) and
// streams the keys in tiles of Bc, staging each K/V tile in SHARED memory. Now all
// Br queries in the block read a given K/V tile from shared memory instead of each
// re-reading it from global — the key trick that makes FlashAttention fast. The
// online-softmax bookkeeping per query row is unchanged.
template <int Br, int Bc>
__global__ void flash_attn_v2(const float* Q, const float* K, const float* V,
                              float* O, int seqlen, float scale) {
    int head = blockIdx.y;
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;                 // 0..Br-1 -> one query row
    int qi = q_start + tid;
    bool valid = qi < seqlen;

    const float* Kh = K + (size_t)head * seqlen * HEAD_DIM;
    const float* Vh = V + (size_t)head * seqlen * HEAD_DIM;

    float qreg[HEAD_DIM], acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;
    if (valid) {
        const float* q = Q + ((size_t)head * seqlen + qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) qreg[d] = q[d];
    }
    float m = -FLT_MAX, l = 0.0f;

    __shared__ float Ks[Bc][HEAD_DIM];
    __shared__ float Vs[Bc][HEAD_DIM];

    int max_key = min(seqlen, q_start + Br);   // causal: no key past the block's last query
    for (int k0 = 0; k0 < max_key; k0 += Bc) {
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += Br) {   // cooperative tile load
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM, kk = k0 + r;
            Ks[r][c] = kk < seqlen ? Kh[(size_t)kk * HEAD_DIM + c] : 0.0f;
            Vs[r][c] = kk < seqlen ? Vh[(size_t)kk * HEAD_DIM + c] : 0.0f;
        }
        __syncthreads();
        if (valid) {
            for (int r = 0; r < Bc; r++) {
                int kj = k0 + r;
                if (kj > qi) break;         // causal mask
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * Ks[r][d];
                s *= scale;
                float m_new = fmaxf(m, s), corr = __expf(m - m_new), p = __expf(s - m_new);
                l = l * corr + p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) acc[d] = acc[d] * corr + p * Vs[r][d];
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (valid) {
        float* o = O + ((size_t)head * seqlen + qi) * HEAD_DIM;
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) o[d] = acc[d] * inv_l;
    }
}

static double check(const float* hO, const float* hQ, const float* hK,
                    const float* hV, int H, int N, int D, float scale);

int main() {
    const int H = 32, N = 2048, D = HEAD_DIM;      // 32 heads, seqlen 2048
    const float scale = 1.0f / sqrtf((float)D);
    const size_t elems = (size_t)H * N * D, bytes = elems * sizeof(float);
    const int iters = 50;

    float *hQ = (float*)malloc(bytes), *hK = (float*)malloc(bytes),
          *hV = (float*)malloc(bytes), *hO = (float*)malloc(bytes);
    srand(0);
    for (size_t i = 0; i < elems; i++) {
        hQ[i] = (float)rand()/RAND_MAX - 0.5f;
        hK[i] = (float)rand()/RAND_MAX - 0.5f;
        hV[i] = (float)rand()/RAND_MAX - 0.5f;
    }
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes); cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    auto time_it = [&](auto launch) {
        launch(); cudaDeviceSynchronize();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int it = 0; it < iters; it++) launch();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms / iters;
    };

    double flops = 2.0 * 2.0 * H * (0.5 * N * N) * D;   // QKᵀ + PV, causal ≈ half
    double naive_scores_gb = (double)H * N * N * 4 / 1e9;

    auto report = [&](const char* name, float ms) {
        cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
        double max_err = check(hO, hQ, hK, hV, H, N, D, scale);
        printf("%-14s time=%.3f ms  %.1f TFLOP/s  max_err=%.2e\n",
               name, ms, flops / (ms / 1e3) / 1e12, max_err);
    };

    int t1 = 128;
    dim3 b1((N + t1 - 1) / t1, H);
    report("v1 per-query", time_it([&]{ flash_attn_v1<<<b1, t1>>>(dQ,dK,dV,dO,N,scale); }));

    constexpr int Br = 64, Bc = 64;
    dim3 b2((N + Br - 1) / Br, H);
    report("v2 tiled-smem", time_it([&]{ flash_attn_v2<Br,Bc><<<b2, Br>>>(dQ,dK,dV,dO,N,scale); }));

    printf("  scores materialized: 0 GB (streamed)   vs naive: %.1f GB\n", naive_scores_gb);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}

// full causal attention in double for a sample of (head, query) rows
static double check(const float* hO, const float* hQ, const float* hK,
                    const float* hV, int H, int N, int D, float scale) {
    double max_err = 0.0;
    int checks[] = {0, 1, 500, 1000, 2047};
    for (int head = 0; head < 4; head++) {
        for (int qi : checks) {
            double mm = -1e30, ll = 0.0, o[D] = {0};
            const float* q = hQ + ((size_t)head*N + qi)*D;
            for (int kj = 0; kj <= qi; kj++) {
                const float* k = hK + ((size_t)head*N + kj)*D;
                double s = 0; for (int d=0; d<D; d++) s += (double)q[d]*k[d];
                s *= scale;
                double mnew = fmax(mm, s), corr = exp(mm-mnew), p = exp(s-mnew);
                ll = ll*corr + p;
                const float* v = hV + ((size_t)head*N + kj)*D;
                for (int d=0; d<D; d++) o[d] = o[d]*corr + p*v[d];
                mm = mnew;
            }
            const float* got = hO + ((size_t)head*N + qi)*D;
            for (int d=0; d<D; d++) {
                double e = fabs(o[d]/ll - got[d]); if (e > max_err) max_err = e;
            }
        }
    }
    return max_err;
}
