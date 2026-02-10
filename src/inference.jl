export predict, embed, cls_pooling, mean_pooling, max_pooling

# ============================================================================
# High-Level Inference API
# ============================================================================

"""
    predict(model, tokenizer, text) -> Matrix{Float32}

Run inference on a single text string.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, 1)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
output = predict(model, tokenizer, "Hello world!")
```
"""
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String)
    m = Flux.testmode!(model)
    input_ids = encode(tokenizer, text)
    input_matrix = reshape(input_ids, :, 1)
    return m(input_matrix)
end

"""
    predict(model, tokenizer, texts; max_length=512) -> Matrix{Float32}

Run batch inference on multiple texts with automatic padding and masking.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `texts::Vector{String}`: Input texts
- `max_length::Int`: Maximum sequence length (default: 512)

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, batch_size)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
output = predict(model, tokenizer, ["Hello world!", "How are you?"])
```
"""
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer,
    texts::Vector{String}; max_length::Int=512)
    m = Flux.testmode!(model)
    input_ids, attention_mask = encode_batch(tokenizer, texts; max_length=max_length)
    return m(input_ids; mask=attention_mask)
end


# ============================================================================
# Pooling Strategies
# ============================================================================

"""
    cls_pooling(output) -> Matrix{Float32}

Extract the [CLS] token representation (first token) from model output.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function cls_pooling(output::AbstractArray{<:Real,3})
    return output[:, 1, :]
end

"""
    mean_pooling(output, attention_mask) -> Matrix{Float32}

Compute mean of token embeddings, weighted by attention mask.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)
- `attention_mask::Matrix{Float32}`: Mask of shape (seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function mean_pooling(output::AbstractArray{<:Real,3}, attention_mask::AbstractMatrix{<:Real})
    # output: (dim, seq_len, batch_size)
    # mask: (seq_len, batch_size) -> expand to (1, seq_len, batch_size)
    mask_expanded = reshape(attention_mask, 1, size(attention_mask)...)

    # Mask the output and sum
    masked_output = output .* mask_expanded
    sum_embeddings = dropdims(sum(masked_output, dims=2), dims=2)  # (dim, batch_size)

    # Count non-padding tokens per batch
    sum_mask = sum(attention_mask, dims=1)  # (1, batch_size)
    sum_mask = max.(sum_mask, 1.0f0)  # Avoid division by zero

    return sum_embeddings ./ sum_mask
end

"""
    max_pooling(output, attention_mask) -> Matrix{Float32}

Compute max of token embeddings, excluding padding tokens.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)
- `attention_mask::Matrix{Float32}`: Mask of shape (seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function max_pooling(output::AbstractArray{<:Real,3}, attention_mask::AbstractMatrix{<:Real})
    # Set padding positions to very negative values so they don't affect max
    mask_expanded = reshape(attention_mask, 1, size(attention_mask)...)
    masked_output = output .* mask_expanded .+ (1.0f0 .- mask_expanded) .* -1.0f9

    return dropdims(maximum(masked_output, dims=2), dims=2)
end


# ============================================================================
# Sentence Embeddings
# ============================================================================

"""
    embed(model, tokenizer, text; pooling=:cls) -> Vector{Float32}

Get sentence embedding for a single text.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text
- `pooling::Symbol`: Pooling strategy - `:cls`, `:mean`, or `:max` (default: `:cls`)

# Returns
- `Vector{Float32}`: Sentence embedding of shape (dim,)
"""
function embed(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String; pooling::Symbol=:cls)
    output = predict(model, tokenizer, text)

    if pooling == :cls
        return vec(cls_pooling(output))
    elseif pooling == :mean
        # For single text, all tokens are valid
        mask = ones(Float32, size(output, 2), 1)
        return vec(mean_pooling(output, mask))
    elseif pooling == :max
        mask = ones(Float32, size(output, 2), 1)
        return vec(max_pooling(output, mask))
    else
        error("Unknown pooling strategy: $pooling. Use :cls, :mean, or :max")
    end
end

"""
    embed(model, tokenizer, texts; pooling=:cls, max_length=512) -> Matrix{Float32}

Get sentence embeddings for multiple texts.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `texts::Vector{String}`: Input texts
- `pooling::Symbol`: Pooling strategy - `:cls`, `:mean`, or `:max` (default: `:cls`)
- `max_length::Int`: Maximum sequence length (default: 512)

# Returns
- `Matrix{Float32}`: Sentence embeddings of shape (dim, batch_size)
"""
function embed(model::DistilBertModel, tokenizer::WordPieceTokenizer, texts::Vector{String};
    pooling::Symbol=:cls, max_length::Int=512)
    m = Flux.testmode!(model)
    input_ids, attention_mask = encode_batch(tokenizer, texts; max_length=max_length)
    output = m(input_ids; mask=attention_mask)

    if pooling == :cls
        return cls_pooling(output)
    elseif pooling == :mean
        return mean_pooling(output, attention_mask)
    elseif pooling == :max
        return max_pooling(output, attention_mask)
    else
        error("Unknown pooling strategy: $pooling. Use :cls, :mean, or :max")
    end
end
