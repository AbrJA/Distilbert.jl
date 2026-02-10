#!/usr/bin/env python3
import os
import sys
import time
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from statistics import median

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def benchmark(func, samples=20, warmup=5):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return median(times)


def run_python_benchmark(model_name="big"):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print("=" * 60)
    print(f"       PYTHON BENCHMARK (models/{model_name})")
    print("=" * 60)

    torch.set_num_threads(4)
    print(f"Torch Threads: {torch.get_num_threads()}")
    print()

    # 1. Load Model
    model = DistilBertModel.from_pretrained(model_path)
    model.eval()
    config = model.config
    vocab_size = config.vocab_size
    print(f"✓ Model loaded: dim={config.dim}, hidden={config.hidden_dim}, layers={config.n_layers}, vocab={vocab_size}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens\n")

    results = {}

    # Tokenizer Benchmark
    print("-" * 60)
    print("BENCHMARK: Tokenizer")
    print("-" * 60)

    s1 = "Hello, this is a test sentence for benchmarking purpose."
    batch_8 = [s1] * 8

    t_short = benchmark(lambda: tokenizer(s1), samples=50)
    print(f"  Short Text (Single):  {t_short:.3f} ms")
    results["Tokenizer (Single)"] = t_short

    t_batch = benchmark(lambda: tokenizer(batch_8, padding=True, truncation=True), samples=50)
    print(f"  Short Text (Batch=8): {t_batch:.3f} ms")
    results["Tokenizer (Batch=8)"] = t_batch
    print()

    # Model Scaling Benchmark
    # Python input shape: (batch_size, seq_len)
    scenarios = [
        {"seq": 32, "batch": 1, "name": "Model (Seq=32, Batch=1)"},
        {"seq": 32, "batch": 8, "name": "Model (Seq=32, Batch=8)"},
        {"seq": 128, "batch": 1, "name": "Model (Seq=128, Batch=1)"},
        {"seq": 128, "batch": 8, "name": "Model (Seq=128, Batch=8)"},
    ]

    for s in scenarios:
        print("-" * 60)
        print(f"BENCHMARK: {s['name']}")
        print("-" * 60)

        # Python: input shape is (batch_size, seq_len)
        input_ids = torch.randint(0, vocab_size, (s["batch"], s["seq"]))

        with torch.no_grad():
            t = benchmark(lambda: model(input_ids))

        results[s["name"]] = t
        print(f"  Median: {t:.2f} ms\n")

    # Summary
    print("=" * 60)
    print(f"PYTHON RESULTS SUMMARY ({model_name})")
    print("=" * 60)
    print(f"Model: dim={config.dim}, hidden={config.hidden_dim}, layers={config.n_layers}")
    print()
    keys_ordered = [
        "Tokenizer (Single)",
        "Tokenizer (Batch=8)",
        "Model (Seq=32, Batch=1)",
        "Model (Seq=32, Batch=8)",
        "Model (Seq=128, Batch=1)",
        "Model (Seq=128, Batch=8)",
    ]
    for k in keys_ordered:
        if k in results:
            print(f"{k:<25}: {results[k]:.2f} ms")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "big"
    run_python_benchmark(model_name)
