# DistilBERT Performance Baseline

**Date:** 2026-02-09
**Model:** Tiny DistilBERT (dim=32, hidden=37, 5 layers, vocab=1124) — loaded from `models/`
**Hardware:** CPU (4 Threads)
**Backends:** Julia uses `MKL.jl` (2 BLAS threads); Python uses `PyTorch` (4 Torch threads)
**Weights:** Both load identical `model.safetensors` / `pytorch_model.bin` from `models/`

## Correctness Validation

```
Input: "DistilBERT is amazing."
Token IDs: ✅ Match (Julia 1-based == Python 0-based + 1)
Max Absolute Difference: 8.34e-07
Verdict: ✅ PASS
```

## Performance Comparison

| Component | Scenario | Julia (MKL) | Python (Torch) | Speedup |
|-----------|----------|-------------|----------------|---------|
| **Tokenizer**| Single | **0.03 ms** | 0.26 ms | **8.6x** 🚀 |
| **Tokenizer**| Batch=8 | **0.23 ms** | 0.94 ms | **4.1x** 🚀 |
| **Model** | Seq=32, B=1 | **0.51 ms** | 6.28 ms | **12.3x** 🚀 |
| **Model** | Seq=32, B=8 | **3.65 ms** | 10.26 ms | **2.8x** 🚀 |
| **Model** | Seq=128, B=1| **6.19 ms** | 8.88 ms | **1.4x** |
| **Model** | Seq=128, B=8| 56.62 ms | **8.85 ms** | 0.16x |

## Key Findings

1. **Julia dominates latency**: For single-item inference (batch=1), Julia is **12x faster** at short sequences and **1.4x faster** at long sequences.
2. **PyTorch dominates batch throughput**: At Seq=128 Batch=8, PyTorch's fused kernels process 8 items in nearly the same time as 1 item (8.85ms vs 8.88ms), while Julia scales linearly.
3. **Tokenizer is much faster**: Julia's native tokenizer is 4-8x faster than HuggingFace's tokenizer.
4. **Numerical parity confirmed**: Outputs match to 7 decimal places.

## Reproduction

```bash
# Run correctness validation
julia --project=. benchmarks/validate_parity.jl

# Run benchmarks (run sequentially, NOT in parallel)
julia --project=. benchmarks/benchmark_julia.jl
./.venv/bin/python3 benchmarks/benchmark_python.py
```
