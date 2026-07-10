# LLM CUDA Kernels — a worklog

Hand-written CUDA kernels for the operations inside an LLM forward pass — each
built up from a naive version, optimized against the bottleneck it exposes, and
**benchmarked honestly on an NVIDIA H100 NVL** with a correctness check against a
reference. Building toward a working **FlashAttention**.

> Sibling to [cuda-softmax-worklog](https://github.com/omkar-droid/cuda-softmax-worklog),
> which is a deep dive on a single kernel (softmax, to 86% of peak HBM). This repo
> is the breadth: the full kernel set an inference engine needs.

The scoreboard depends on the kernel. Elementwise/reduction kernels are
**memory-bound**, so the metric is **% of peak HBM bandwidth** (H100 NVL ≈ 3.9 TB/s)
with `torch` as the reference to beat or match. Compute-bound kernels (matmul,
attention) are measured in TFLOP/s and honest % of cuBLAS / FlashInfer.

## Kernels

| # | Kernel | Status | Result |
|---|---|---|---|
| 1 | **SiLU** (`silu.cu`) | ✅ | 89% of peak HBM (float4), correct to 3e-7 |
| 2 | RMSNorm | 🚧 next | — |
| 3 | LayerNorm | ⏳ | — |
| 4 | KV-cache update (scatter) | ⏳ | — |
| 5 | Top-k | ⏳ | — |
| 6 | Tiled MatMul | ⏳ | vs cuBLAS (learning GEMM) |
| 7 | **FlashAttention** | ⏳ | the capstone |

## Results (H100 NVL, fp32)

**SiLU** — `out = x * sigmoid(x)`, memory-bound (read x, write out):

| version | idea | bandwidth | % of peak |
|---|---|---:|---:|
| v1 scalar | one float per thread | 2,018 GB/s | 52% |
| **v2 float4** | four floats per thread (wide, aligned loads) | **3,484 GB/s** | **89%** |

## Build & run

```bash
make            # builds every src/*.cu into bin/
./bin/silu
```

Requires an NVIDIA GPU (compute capability ≥ 8.0) and the CUDA toolkit.
Kernels target `sm_90` (Hopper); change `-arch` in the Makefile for other GPUs.
