# DistilBERT.jl

A high-performance, pure Julia implementation of the **DistilBERT** model, built with **Flux.jl**.

This project implements the DistilBERT architecture from scratch, featuring a custom WordPiece tokenizer and compatibility with Hugging Face pre-trained weights (`.safetensors`). It has been optimized for performance, achieving near-parity with PyTorch on CPU for large models and significantly outperforming it for small-batch inference.

## 🚀 Features

- **Pure Julia**: No Python dependencies required for inference.
- **High Performance**: Optimized attention mechanism using `NNlib.dot_product_attention`.
- **Hugging Face Compatible**:
    - Loads weights directly from `model.safetensors`.
    - Implements `WordPieceTokenizer` matching `BertTokenizer` logic.
- **Task-Specific Heads**:
    - `DistilBertModel` (Base)
    - `DistilBertForSequenceClassification`
    - `DistilBertForTokenClassification`
    - `DistilBertForQuestionAnswering`
- **Verified Accuracy**: Validated against Python's `transformers` library with `< 1e-4` numerical difference.

## 📦 Installation

This package is currently a local project. Clone the repository and instantiate the environment:

```bash
git clone https://github.com/StartYourStart/Distilbert.jl.git
cd Distilbert.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## ⚡ Quick Start

### 1. Download Model Weights
You need a folder containing `config.json`, `vocab.txt`, and `model.safetensors` from a DistilBERT model (e.g., `distilbert-base-uncased`).

### 2. Basic Inference
```julia
using Distilbert

# Load model and tokenizer
model_path = "path/to/your/model_directory"
model = load_model(model_path)
tokenizer = WordPieceTokenizer(joinpath(model_path, "vocab.txt"))

# Run inference on a single sentence
text = "DistilBERT is amazing in Julia!"
output = inference(model, tokenizer, text)

println("Output shape: ", size(output))
# (dim, seq_len, 1)
```

### 3. Sentence Embeddings
Easily extract pooled embeddings for downstream tasks:

```julia
# Get [CLS] token embedding
cls_embedding = embed(model, tokenizer, text; pooling=:cls)

# Get Mean-pooled embedding
mean_embedding = embed(model, tokenizer, text; pooling=:mean)

println("Embedding size: ", size(cls_embedding))
# (dim,)
```

### 4. Batch Processing
Efficiently process multiple sentences with automatic padding and masking:

```julia
texts = [
    "Julia is fast.",
    "Machine learning is exciting.",
    "This is a longer sentence to test padding."
]

# Get batch embeddings
batch_embeddings = embed(model, tokenizer, texts; pooling=:mean)

println("Batch shape: ", size(batch_embeddings))
# (dim, batch_size)
```

## 📊 Benchmarks

Hardware: Linux, 4 Threads. Comparison vs PyTorch (Hugging Face).

### Small Model (dim=32)
| Task | Julia (MKL) | Python | Speedup |
|------|-------------|--------|---------|
| **Tokenizer** (Single) | **0.04 ms** | 0.16 ms | **4.0x** 🚀 |
| **Inference** (Batch=1) | **0.53 ms** | 6.05 ms | **11.4x** 🚀 |
| **Inference** (Batch=8) | **4.67 ms** | 4.78 ms | **1.02x** |

### Big Model (Standard DistilBERT)
| Task | Julia (MKL) | Python | Speedup |
|------|-------------|--------|---------|
| **Tokenizer** (Single) | **0.01 ms** | 0.10 ms | **10.0x** 🚀 |
| **Inference** (Batch=1) | 55.70 ms | **46.45 ms** | 0.83x |
| **Inference** (Batch=8) | 360.22 ms | **230.06 ms** | 0.64x |

## ✅ Verification

To ensure the implementation matches the reference PyTorch implementation, run the parity validation script:

```bash
# Verify against a 'small' model (fast)
julia --project=. benchmarks/validate_parity.jl small

# Verify against a 'big' model
julia --project=. benchmarks/validate_parity.jl big
```

**Requirements:**
- A Python environment (`.venv`) with `torch` and `transformers` installed is required for running the comparison scripts.
- To use MKL for best performance in Julia: `using MKL` (ensure it's in your project dependencies).

## Project Structure

- `src/Distilbert.jl`: Core model architecture and high-level API.
- `src/Tokenizer.jl`: WordPiece tokenizer implementation.
- `benchmarks/`: Performance benchmarking scripts (`benchmark_julia.jl`, `benchmark_python.py`).
- `models/`: Directory structure for storing downloaded model weights.
