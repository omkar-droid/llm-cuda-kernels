// RMSNorm:  out[r,i] = x[r,i] / sqrt( mean_i(x[r,i]^2) + eps ) * w[i]
// over an [R, D] matrix (one row = one token's hidden vector).
//
// The first *reduction* kernel: to normalize one element a thread needs the sum
// of squares over its whole row, but the row is spread across BLOCK threads — so
// the threads must cooperate to combine BLOCK partial sums into one. Two ways:
//   v1  shared-memory tree reduction, scalar loads   (the clear version)
//   v2  warp-shuffle reduction + float4 loads         (the fast version)
// Memory-bound: metric is % of peak HBM (read x, write out).
//
// Build & run:  nvcc -O3 -arch=sm_90 src/rmsnorm.cu -o rmsnorm && ./rmsnorm

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK 256

// ---------- v1: shared-memory tree reduction ----------
__global__ void rmsnorm_v1(const float* x, const float* w, float* out, int D, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)row * D;
    float* outr = out + (size_t)row * D;

    // 1) each thread sums the squares of the elements it owns (strided)
    float partial = 0.f;
    for (int i = tid; i < D; i += BLOCK) partial += xr[i] * xr[i];

    // 2) combine BLOCK partials into one — the reduction.
    //    each step, the lower half of threads absorbs the upper half.
    __shared__ float s[BLOCK];
    s[tid] = partial;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();                     // everyone waits before the next step
    }
    // s[0] now holds the whole row's sum of squares (all threads can read it)
    float inv_rms = rsqrtf(s[0] / D + eps);

    // 3) normalize
    for (int i = tid; i < D; i += BLOCK) outr[i] = xr[i] * inv_rms * w[i];
}

// ---------- v2: warp-shuffle reduction + float4 ----------
__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;                                 // 32 threads -> lane 0 holds the sum, in 5 steps
}

__global__ void rmsnorm_v2(const float4* x, const float4* w, float4* out, int D4, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float4* xr = x + (size_t)row * D4;
    float4* outr = out + (size_t)row * D4;

    float partial = 0.f;
    for (int i = tid; i < D4; i += BLOCK) {
        float4 v = xr[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // reduce within each warp (no shared mem), then across the 8 warps via shared
    partial = warp_sum(partial);
    __shared__ float s[BLOCK / 32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) s[wid] = partial;
    __syncthreads();
    if (wid == 0) {
        partial = (lane < BLOCK / 32) ? s[lane] : 0.f;
        partial = warp_sum(partial);
        if (lane == 0) s[0] = partial;
    }
    __syncthreads();
    float inv_rms = rsqrtf(s[0] / (D4 * 4) + eps);

    for (int i = tid; i < D4; i += BLOCK) {
        float4 v = xr[i], wv = w[i], r;
        r.x = v.x * inv_rms * wv.x; r.y = v.y * inv_rms * wv.y;
        r.z = v.z * inv_rms * wv.z; r.w = v.w * inv_rms * wv.w;
        outr[i] = r;
    }
}

#define TIME(call, ms) do { \
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); \
    call; cudaDeviceSynchronize(); cudaEventRecord(a); \
    for (int it = 0; it < iters; it++) { call; } \
    cudaEventRecord(b); cudaEventSynchronize(b); \
    cudaEventElapsedTime(&ms, a, b); ms /= iters; } while (0)

int main() {
    const int R = 32768, D = 4096;           // 32k tokens, Mistral-size hidden
    const size_t n = (size_t)R * D, bytes = n * sizeof(float);
    const float eps = 1e-5f;
    const int iters = 100;
    const double peak = 3900.0;              // H100 NVL HBM3 ~3.9 TB/s

    float *h_x = (float*)malloc(bytes), *h_w = (float*)malloc(D * 4), *h_out = (float*)malloc(bytes);
    for (size_t i = 0; i < n; i++) h_x[i] = (float)((i % 511) - 255) / 128.0f;
    for (int i = 0; i < D; i++) h_w[i] = 1.0f + (i % 7) * 0.01f;

    float *d_x, *d_w, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_w, D * 4); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, D * 4, cudaMemcpyHostToDevice);

    auto check = [&](const char* name, float ms) {
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int r = 0; r < 64; r++) {                       // spot-check 64 rows
            double ss = 0.0;
            for (int i = 0; i < D; i++) { double v = h_x[(size_t)r*D+i]; ss += v*v; }
            double inv = 1.0 / sqrt(ss / D + eps);
            for (int i = 0; i < D; i++) {
                double ref = h_x[(size_t)r*D+i] * inv * h_w[i];
                double e = fabs(ref - h_out[(size_t)r*D+i]); if (e > max_err) max_err = e;
            }
        }
        double bw = 2.0 * bytes / 1e9 / (ms / 1e3);
        printf("%-12s time=%.3f ms  bw=%.0f GB/s  (%2.0f%% of peak)  max_err=%.2e\n",
               name, ms, bw, 100.0 * bw / peak, max_err);
    };

    float ms;
    TIME((rmsnorm_v1<<<R, BLOCK>>>(d_x, d_w, d_out, D, eps)), ms);
    check("v1 shared", ms);
    TIME((rmsnorm_v2<<<R, BLOCK>>>((float4*)d_x, (float4*)d_w, (float4*)d_out, D/4, eps)), ms);
    check("v2 shuffle", ms);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out); free(h_x); free(h_w); free(h_out);
    return 0;
}
