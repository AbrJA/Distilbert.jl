struct Embeddings
    word_embeddings::Embedding
    position_embeddings::Embedding
    LayerNorm::LayerNorm
    dropout::Dropout
end

Flux.@layer Embeddings

function Embeddings(config::DistilBertConfig)
    return Embeddings(
        Embedding(config.vocab_size => config.dim),
        Embedding(config.max_position_embeddings => config.dim),
        LayerNorm(config.dim; eps=config.layer_norm_eps),
        Dropout(config.dropout)
    )
end

function (m::Embeddings)(input_ids::AbstractMatrix{<:Integer})
    seq_length = size(input_ids, 1)

    words_embeddings = m.word_embeddings(input_ids) # (dim, seq_len, batch_size)

    # Position embeddings: (dim, seq_len) -> broadcast to (dim, seq_len, batch_size)
    # We use 1:seq_length for Julia indices
    pos_ids = 1:seq_length
    position_embeddings = m.position_embeddings(pos_ids) # (dim, seq_len)

    embeddings = words_embeddings .+ position_embeddings
    embeddings = m.LayerNorm(embeddings)
    embeddings = m.dropout(embeddings)

    return embeddings
end

struct MultiHeadSelfAttention
    n_heads::Int
    dim::Int
    head_dim::Int
    q_lin::Dense
    k_lin::Dense
    v_lin::Dense
    out_lin::Dense
    dropout::Dropout
end

Flux.@layer MultiHeadSelfAttention

function MultiHeadSelfAttention(config::DistilBertConfig)
    head_dim = config.dim ÷ config.n_heads
    return MultiHeadSelfAttention(
        config.n_heads,
        config.dim,
        head_dim,
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dropout(config.dropout)
    )
end

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Real,3}; mask::AbstractArray=ones(Float32, 0, 0))
    # x shape: (dim, seq_len, batch_size)

    q = m.q_lin(x)
    k = m.k_lin(x)
    v = m.v_lin(x)

    # Perform dot product attention using NNlib
    mask_nnlib = nothing
    if length(mask) > 0
        mask_nnlib = reshape(mask, size(mask, 1), 1, 1, size(mask, 2)) .== 1
    end

    # Apply attention
    # fdrop is applied to attention weights after softmax
    context, attn_weights = dot_product_attention(q, k, v;
        nheads=m.n_heads,
        mask=mask_nnlib,
        fdrop=m.dropout)

    # context is (dim, seq_len, batch_size)

    output = m.out_lin(context)
    return output
end

struct FeedForward
    lin1::Dense
    lin2::Dense
    dropout::Dropout
end

Flux.@layer FeedForward

function FeedForward(config::DistilBertConfig)
    return FeedForward(
        Dense(config.dim => config.hidden_dim, gelu),
        Dense(config.hidden_dim => config.dim),
        Dropout(config.dropout)
    )
end

function (m::FeedForward)(x::AbstractArray{<:Real,3})
    return m.lin2(m.dropout(m.lin1(x)))
end


struct TransformerBlock
    attention::MultiHeadSelfAttention
    sa_layer_norm::LayerNorm
    ffn::FeedForward
    output_layer_norm::LayerNorm
end

Flux.@layer TransformerBlock

function TransformerBlock(config::DistilBertConfig)
    return TransformerBlock(
        MultiHeadSelfAttention(config),
        LayerNorm(config.dim; eps=config.layer_norm_eps),
        FeedForward(config),
        LayerNorm(config.dim; eps=config.layer_norm_eps)
    )
end

function (m::TransformerBlock)(x::AbstractArray{<:Real,3}; mask::AbstractMatrix{Float32}=ones(Float32, 0, 0))
    # Self-Attention
    sa_output = m.attention(x; mask=mask)
    sa_output = m.sa_layer_norm(sa_output .+ x)

    # Feed Forward
    ffn_output = m.ffn(sa_output)
    output = m.output_layer_norm(ffn_output .+ sa_output)

    return output
end
