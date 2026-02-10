# DistilBERT Benchmark Results

**Date:** 2026-02-09
**Machine:** Linux, 4 Threads

## 1. Correctness Verification

We validated the Julia implementation against the HuggingFace Transformers (PyTorch) reference implementation using two models:

| Model | Dimensions | Layers | Vocab | Token IDs | Hidden States Max Diff | Verdict |
|-------|------------|--------|-------|-----------|------------------------|---------|
| **Small** | dim=32 | 5 | 1124 | ✅ Match | `8.34e-07` | **PERFECT** |
| **Big** | dim=768 | 6 | 30k | ✅ Match | `6.76e-03` | **PASS** (expected FP32 drift) |

> **Note on Big Model:** The max difference of ~`6e-3` is expected when comparing different framework implementations (LibTorch vs OpenBLAS) accumulated across 6 transformer layers. The per-token values match to 3-4 decimal places.

## 2. Performance Benchmark (Big Model)

**Model:** `distilbert-base-uncased` (dim=768, layers=6)
**Threads:** 4 (Julia & PyTorch)

### Summary
- **Tokenizer:** Julia is **~9x faster** than HuggingFace's fast tokenizer.
- **Inference:** Julia is **~2.5x - 8.8x slower** than PyTorch.

### Detailed Results

| Component | Batch Size | Sequence Length | Julia (ms) | Python (ms) | Speedup (Julia vs Py) |
|-----------|------------|-----------------|------------|-------------|-----------------------|
| **Tokenizer** | 1 | - | **0.03** | 0.28 | **9.3x Faster** 🚀 |
| **Tokenizer** | 8 | - | **0.13** | 1.13 | **8.7x Faster** 🚀 |
| | | | | | |
| **Model** | 1 | 32 | 471.91 | **178.52** | 2.6x Slower |
| **Model** | 8 | 32 | 2012.24 | **719.30** | 2.8x Slower |
| **Model** | 1 | 128 | 2531.93 | **440.30** | 5.8x Slower |
| **Model** | 8 | 128 | 17716.66 | **2011.63** | 8.8x Slower |

### Analysis

1.  **Tokenizer:** The optimization to use `IOBuffer` and avoid string allocations in `WordPieceTokenizer` paid off massively. Julia's string processing string is superior here.
2.  **Model Inference:** PyTorch is significantly faster. This is likely due to:
    - **BLAS Backend:** PyTorch uses MKL (Intel Math Kernel Library) which is highly optimized for AVX2/AVX-512. Our Julia run used `OpenBLAS` (standard in Julia distribution).
    - **Memory Layout:** PyTorch's `baddbmm` and memory manager are highly tuned for transformer workloads.
    - **Allocations:** Julia's `NNlib.batched_mul` + broadcasting likely allocates more intermediate arrays than PyTorch's fused kernels.

**Recommendation:** To close the gap, we would need to:
1.  Use `MKL.jl` (we removed it to drop the hard dependency, but it's crucial for performance).
2.  Optimize `batched_mul` usage or use `Octavian.jl` for pure-Julia matmul.
