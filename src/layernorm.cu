// LayerNorm:  out[r,i] = (x[r,i] - mean) / sqrt(var + eps) * w[i] + b[i]
//   mean = (1/D) Σ x,   var = (1/D) Σ x² - mean²   (both over the row)
//
// RMSNorm needed ONE reduction (Σx²). LayerNorm needs TWO — the mean and the
// variance — but we get both in a single pass by accumulating Σx and Σx²
// together and reducing the pair. Same shared-memory tree reduction that won for
// RMSNorm (float4 lost there, so we don't bother). Memory-bound: % of peak HBM.
//
// Build & run:  nvcc -O3 -arch=sm_90 src/layernorm.cu -o layernorm && ./layernorm

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void layernorm(const float* x, const float* w, const float* b,
                          float* out, int D, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)row * D;
    float* outr = out + (size_t)row * D;

    // 1) one pass: each thread accumulates Σx and Σx² over its strided elements
    float s = 0.f, s2 = 0.f;
    for (int i = tid; i < D; i += BLOCK) { float v = xr[i]; s += v; s2 += v * v; }

    // 2) reduce BOTH sums together (one tree, two accumulators)
    __shared__ float ssum[BLOCK], ssq[BLOCK];
    ssum[tid] = s; ssq[tid] = s2;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { ssum[tid] += ssum[tid + stride]; ssq[tid] += ssq[tid + stride]; }
        __syncthreads();
    }
    // 3) mean, variance, normalizer (all threads read the shared totals)
    float mean = ssum[0] / D;
    float var  = ssq[0] / D - mean * mean;
    float inv  = rsqrtf(var + eps);

    // 4) normalize + scale + shift
    for (int i = tid; i < D; i += BLOCK) outr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

#define TIME(call, ms) do { \
    cudaEvent_t a, b_; cudaEventCreate(&a); cudaEventCreate(&b_); \
    call; cudaDeviceSynchronize(); cudaEventRecord(a); \
    for (int it = 0; it < iters; it++) { call; } \
    cudaEventRecord(b_); cudaEventSynchronize(b_); \
    cudaEventElapsedTime(&ms, a, b_); ms /= iters; } while (0)

int main() {
    const int R = 32768, D = 4096;
    const size_t n = (size_t)R * D, bytes = n * sizeof(float);
    const float eps = 1e-5f;
    const int iters = 100;
    const double peak = 3900.0;

    float *h_x = (float*)malloc(bytes), *h_w = (float*)malloc(D*4),
          *h_b = (float*)malloc(D*4), *h_out = (float*)malloc(bytes);
    for (size_t i = 0; i < n; i++) h_x[i] = (float)((i % 511) - 255) / 128.0f;
    for (int i = 0; i < D; i++) { h_w[i] = 1.0f + (i % 7) * 0.01f; h_b[i] = (i % 5) * 0.02f; }

    float *d_x, *d_w, *d_b, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_w, D*4); cudaMalloc(&d_b, D*4); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, D*4, cudaMemcpyHostToDevice);

    float ms;
    TIME((layernorm<<<R, BLOCK>>>(d_x, d_w, d_b, d_out, D, eps)), ms);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (int r = 0; r < 64; r++) {
        double sm = 0, sq = 0;
        for (int i = 0; i < D; i++) { double v = h_x[(size_t)r*D+i]; sm += v; sq += v*v; }
        double mean = sm/D, var = sq/D - mean*mean, inv = 1.0/sqrt(var+eps);
        for (int i = 0; i < D; i++) {
            double ref = (h_x[(size_t)r*D+i]-mean)*inv*h_w[i]+h_b[i];
            double e = fabs(ref - h_out[(size_t)r*D+i]); if (e > max_err) max_err = e;
        }
    }
    double bw = 2.0 * bytes / 1e9 / (ms / 1e3);
    printf("layernorm    time=%.3f ms  bw=%.0f GB/s  (%2.0f%% of peak)  max_err=%.2e\n",
           ms, bw, 100.0 * bw / peak, max_err);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    free(h_x); free(h_w); free(h_b); free(h_out);
    return 0;
}
