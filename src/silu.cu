// SiLU (swish):  out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
//
// A memory-bound elementwise kernel — the math is trivial, so all that matters
// is how fast we read x and write out from HBM. Worklog:
//   v1  scalar   — one float per thread          (baseline)
//   v2  float4   — four floats per thread (wide, aligned loads)
//
// Build & run:  nvcc -O3 -arch=sm_90 src/silu.cu -o silu && ./silu

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// v1: one element per thread
__global__ void silu_v1(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu(x[i]);
}

// v2: four elements per thread via a single 16-byte load/store
__global__ void silu_v2(const float4* x, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i];                       // one wide load = 4 floats
        v.x = silu(v.x); v.y = silu(v.y);
        v.z = silu(v.z); v.w = silu(v.w);
        out[i] = v;                            // one wide store
    }
}

float bench(void (*launch)(const float*, float*, int), const float* d_x,
            float* d_out, int N) {
    launch(d_x, d_out, N);                      // warm up
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = 200;
    cudaEventRecord(a);
    for (int it = 0; it < iters; it++) launch(d_x, d_out, N);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms / iters;
}

void run_v1(const float* x, float* out, int N) {
    int t = 256, blk = (N + t - 1) / t;
    silu_v1<<<blk, t>>>(x, out, N);
}
void run_v2(const float* x, float* out, int N) {
    int n4 = N / 4, t = 256, blk = (n4 + t - 1) / t;
    silu_v2<<<blk, t>>>((const float4*)x, (float4*)out, n4);
}

int main() {
    const int N = 1 << 26;
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_x = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_x[i] = (float)((i % 2000) - 1000) / 250.0f;

    float *d_x, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    double peak = 3900.0;                        // H100 NVL HBM3 ~3.9 TB/s (measured ceiling)
    struct { const char* name; void (*fn)(const float*, float*, int); } kernels[] = {
        {"v1 scalar", run_v1}, {"v2 float4", run_v2},
    };
    for (auto& k : kernels) {
        float ms = bench(k.fn, d_x, d_out, N);
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double ref = (double)h_x[i] / (1.0 + exp(-(double)h_x[i]));
            double e = fabs(ref - h_out[i]); if (e > max_err) max_err = e;
        }
        double bw = 2.0 * bytes / 1e9 / (ms / 1e3);
        printf("%-10s  time=%.3f ms  bw=%.0f GB/s  (%2.0f%% of peak)  max_err=%.2e\n",
               k.name, ms, bw, 100.0 * bw / peak, max_err);
    }
    cudaFree(d_x); cudaFree(d_out); free(h_x); free(h_out);
    return 0;
}
