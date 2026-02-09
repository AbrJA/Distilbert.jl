# DistilBERT Benchmark: Julia vs Python (PyTorch CPU)

**Date:** 2026-02-09
**Environment:** Linux (4 cores)
**Optimization Level:** LoopVectorization (4D Kernels), Single-Threaded `@turbo`
**Model Config:** DistilBERT Base (dim=768, heads=12, layers=6)

## Summary

Julia demonstrates **superior latency** for single-item inference and tokenization, making it ideal for real-time applications. PyTorch (CPU) retains an advantage in throughput for larger batches, likely due to highly optimized MKL/BLAS kernels.

| Operation | Batch Size | Julia (ms) | Python (ms) | Speedup (Julia vs PyTorch) |
|-----------|------------|------------|-------------|----------------------------|
| **Tokenizer** | 1 | **0.02** | 0.10 | **5.0x Faster** 🚀 |
| **Tokenizer** | 8 | **0.16** | 1.54 | **9.6x Faster** 🚀 |
| **Forward Pass** | 1 (Seq 32) | **0.56** | 4.12 | **7.3x Faster** 🚀 |
| **Forward Pass** | 8 (Seq 128) | 113.00 | 12.34 | 0.11x (9x Slower) |
| **Embeddings** | 8 (Seq 128) | 28.64 | 7.54 | 0.26x (4x Slower) |
| **End-to-End** | 1 (Seq 32) | **0.85** | 4.42 | **5.2x Faster** 🚀 |

## Details

### 1. Tokenization
Julia's `WordPieceTokenizer` is extremely efficient, utilizing string views and minimal allocations. It consistently outperforms the Python implementation.

### 2. Single-Item Inference (Latency)
With `LoopVectorization.jl` applying AVX-512 optimizations on custom 4D attention kernels (avoiding permutations), Julia achieves sub-millisecond inference (0.56ms), significantly beating PyTorch overhead.

### 3. Batch Throughput
For larger matrices (Batch=8, Seq=128), PyTorch's backend (MKL/OpenMP) efficiently parallelizes across cores.
- **Current Bottleneck:** Attention mechanism scale (128x128 matrices are small for BLAS but large enough for overhead).
- **Optimization Attempted:** Multi-threading with `@tturbo` showed slight regression (126ms vs 113ms), indicating thread overhead outweighs parallelization gains at this scale on 4 cores.
- **Future Work:** GPU support (`CUDA.jl` / `Metal.jl`) or further blocking optimizations could close this gap.

## Methodology
- **Julia:** `benchmark/benchmark_julia.jl` (using `BenchmarkTools.jl`)
- **Python:** `benchmark/benchmark_python.py` (using `time.perf_counter` warm-up loops)
- **Precision:** Float32
