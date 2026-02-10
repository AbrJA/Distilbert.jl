#!/usr/bin/env julia
#=
Parity Validation: Julia vs Python
Loads the SAME model from models/, feeds IDENTICAL token IDs,
and compares hidden state outputs element-wise.
Supports both small and big models.
=#
using Distilbert
using Flux
using JSON
using Printf

const MODELS_DIR = joinpath(dirname(@__DIR__), "models")

function validate_parity(model_name::String)
    model_path = joinpath(MODELS_DIR, model_name)
    if !isdir(model_path)
        error("Model directory not found: $model_path")
    end

    println("="^60)
    println("  PARITY VALIDATION: Julia vs Python ($model_name)")
    println("="^60)

    # 1. Load Julia Model
    println("\n[Julia] Loading model from models/$model_name/...")
    model = load_model(model_path)
    m = Flux.testmode!(model)
    config = model.config
    println("[Julia] Model: dim=$(config.dim), hidden=$(config.hidden_dim), layers=$(config.n_layers), vocab=$(config.vocab_size)")

    # Tokenizer
    vocab_path = joinpath(model_path, "vocab.txt")
    tokenizer = WordPieceTokenizer(vocab_path; do_lower_case=true)

    # 2. Test Text
    text = "DistilBERT is amazing."
    println("\nTest text: \"$text\"")

    # 3. Run Julia
    julia_ids = encode(tokenizer, text)
    println("[Julia] Token IDs (1-based): $julia_ids")

    input_matrix = reshape(julia_ids, :, 1)  # (seq_len, 1)
    julia_output = m(input_matrix)            # (dim, seq_len, 1)
    jl_out = dropdims(julia_output, dims=3)   # (dim, seq_len)
    println("[Julia] Output shape: $(size(jl_out))")

    # 4. Run Python (inline script)
    println("\n[Python] Running reference model...")
    python_executable = joinpath(dirname(@__DIR__), ".venv/bin/python3")
    if !isfile(python_executable)
        python_executable = "python3"
    end

    python_script = """
import torch, json, os, sys
from transformers import DistilBertModel, DistilBertTokenizer

model_path = "$(model_path)"
text = "$(text)"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertModel.from_pretrained(model_path)
model.eval()

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

result = {
    "input_ids": input_ids.tolist()[0],
    "last_hidden_state": last_hidden_state.tolist()[0]
}
print(json.dumps(result))
"""

    cmd = `$python_executable -c $python_script`
    python_out = try
        read(cmd, String)
    catch e
        println("ERROR: Python failed: $e")
        return false
    end

    py_results = JSON.parse(python_out)
    py_input_ids = Int.(py_results["input_ids"])
    # Python output shape: list of [seq_len][dim] -> transpose to (dim, seq_len)
    py_hidden = hcat([Float32.(row) for row in py_results["last_hidden_state"]]...)

    println("[Python] Token IDs (0-based): $py_input_ids")
    println("[Python] Output shape: $(size(py_hidden))")

    # Julia: 1-based, Python: 0-based
    ids_match = (julia_ids .- 1) == py_input_ids
    if ids_match
        println("  ✅ Token IDs match")
    else
        println("  ❌ Token IDs MISMATCH!")
        println("  Julia (1-based): $julia_ids")
        println("  Python (0-based): $py_input_ids")
    end

    println("\n" * "-"^60)
    println("CHECK 2: Hidden State Numerical Parity")
    println("-"^60)

    diff = abs.(jl_out .- py_hidden)
    max_diff = maximum(diff)
    mean_diff = sum(diff) / length(diff)

    @printf("  Max  Absolute Difference: %.2e\n", max_diff)
    @printf("  Mean Absolute Difference: %.2e\n", mean_diff)

    if max_diff < 1e-4
        println("  ✅ PASS — outputs match within 1e-4 tolerance")
    elseif max_diff < 1e-3
        println("  ⚠️  MARGINAL — outputs within 1e-3 but not 1e-4")
    else
        println("  ❌ FAIL — outputs differ significantly")
    end

    println("\n" * "-"^60)
    println("Per-Token Comparison (first 3 dims)")
    println("-"^60)
    seq_len = size(jl_out, 2)
    ndims_show = min(3, config.dim)
    for t in 1:min(seq_len, 5)
        jl_vals = jl_out[1:ndims_show, t]
        py_vals = py_hidden[1:ndims_show, t]
        if ndims_show >= 3
            @printf("  Token %d: Julia=[%.4f, %.4f, %.4f]  Python=[%.4f, %.4f, %.4f]\n",
                t, jl_vals[1], jl_vals[2], jl_vals[3],
                py_vals[1], py_vals[2], py_vals[3])
        end
    end

    println("\n" * "="^60)
    println("VERDICT ($model_name)")
    println("="^60)
    all_pass = ids_match && max_diff < 1e-4
    if all_pass
        println("  ✅ ALL CHECKS PASSED")
    else
        println("  ❌ SOME CHECKS FAILED")
    end

    return all_pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    model_name = length(ARGS) > 0 ? ARGS[1] : "small"
    validate_parity(model_name)
end
