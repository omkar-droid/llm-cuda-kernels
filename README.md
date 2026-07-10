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
| 2 | **RMSNorm** (`rmsnorm.cu`) | ✅ | 82% of peak HBM — **ties vLLM's fused kernel**, 4.2× `torch` |
| 3 | **LayerNorm** (`layernorm.cu`) | ✅ | 80% of peak HBM — 1.2× `torch.layer_norm` |
| 4 | KV-cache update (scatter) | 🚧 next | — |
| 5 | Top-k | ⏳ | — |
| 6 | Tiled MatMul | ⏳ | vs cuBLAS (learning GEMM) |
| 7 | **FlashAttention** | ⏳ | the capstone |

## Results (H100 NVL, fp32)

**SiLU** — `out = x * sigmoid(x)`, memory-bound (read x, write out):

| version | idea | bandwidth | % of peak |
|---|---|---:|---:|
| v1 scalar | one float per thread | 2,018 GB/s | 52% |
| **v2 float4** | four floats per thread (wide, aligned loads) | **3,484 GB/s** | **89%** |

**RMSNorm** — normalize each row by its RMS; a *reduction* (threads cooperate to
sum `x²` over a row). `[32768, 4096]`, fp32:

| version | idea | bandwidth | % of peak |
|---|---|---:|---:|
| **v1 shared** | shared-memory tree reduction, scalar loads | **3,217 GB/s** | **82%** |
| v2 shuffle | warp-shuffle reduction + float4 | 2,937 GB/s | 75% |
| vLLM fused `rms_norm` (fp16) | production hand-tuned kernel | — | **82%** |
| `torch.nn.functional.rms_norm` | unfused reference | 768 GB/s | 20% |

Two honest notes:
- The "fancier" v2 (warp-shuffle + float4) came out **slower** than v1 — the reduction
  was never the bottleneck (memory was), and v1 already sits near the HBM ceiling, so
  the extra machinery only added overhead. Measure, don't assume.
- **vs vLLM:** for a memory-bound kernel, *% of peak HBM is the ceiling* — you can't beat
  the memory bus. Ours and vLLM's fused kernel both hit **82%**, i.e. we're at the same
  place vLLM is. (vLLM runs fp16 so its wall-clock is ~2× lower — it moves half the bytes —
  but the *efficiency* is identical.) The comparison that leaves real room is the
  compute-bound kernels (matmul vs cuBLAS, attention vs FlashInfer) — those come later.

**LayerNorm** — two reductions (mean + variance) in one pass. `[32768, 4096]`, fp32:

| version | bandwidth | % of peak |
|---|---:|---:|
| **ours** | **3,135 GB/s** | **80%** |
| `torch.nn.functional.layer_norm` (fused) | 2,541 GB/s | 65% |

## Build & run

```bash
make            # builds every src/*.cu into bin/
./bin/silu
```

Requires an NVIDIA GPU (compute capability ≥ 8.0) and the CUDA toolkit.
Kernels target `sm_90` (Hopper); change `-arch` in the Makefile for other GPUs.
