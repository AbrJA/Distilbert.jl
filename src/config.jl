export DistilBertConfig

"""
    DistilBertConfig

Configuration for the DistilBERT model.

# Fields
- `vocab_size::Int`: Vocabulary size (default: 30522)
- `dim::Int`: Dimensionality of the encoder layers and the pooler layer (default: 768)
- `n_layers::Int`: Number of hidden layers in the Transformer encoder (default: 6)
- `n_heads::Int`: Number of attention heads for each attention layer in the Transformer encoder (default: 12)
- `hidden_dim::Int`: Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder (default: 3072)
- `dropout::Float32`: The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler (default: 0.1)
- `max_position_embeddings::Int`: The maximum sequence length that this model might ever be used with (default: 512)
- `initializer_range::Float32`: The standard deviation of the truncated_normal_initializer for initializing all weight matrices (default: 0.02)
- `qa_dropout::Float32`: Dropout probability for the QA head (default: 0.1)
- `seq_classif_dropout::Float32`: Dropout probability for the sequence classification head (default: 0.2)
"""
@kwdef struct DistilBertConfig
    vocab_size::Int = 30522
    dim::Int = 768
    n_layers::Int = 6
    n_heads::Int = 12
    hidden_dim::Int = 3072
    dropout::Float32 = 0.1f0
    max_position_embeddings::Int = 512
    initializer_range::Float32 = 0.02f0
    qa_dropout::Float32 = 0.1f0
    seq_classif_dropout::Float32 = 0.2f0
    layer_norm_eps::Float32 = 1f-12
end

function Base.show(io::IO, c::DistilBertConfig)
    print(io, "DistilBertConfig(dim=$(c.dim), layers=$(c.n_layers), heads=$(c.n_heads), hidden=$(c.hidden_dim), vocab=$(c.vocab_size))")
end
