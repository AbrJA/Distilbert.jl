module Tokenizer

using Unicode

export WordPieceTokenizer, tokenize, encode, encode_batch, load_vocab

struct WordPieceTokenizer
    vocab::Dict{String,Int}
    ids_to_tokens::Dict{Int,String}
    unk_token::String
    sep_token::String
    pad_token::String
    cls_token::String
    mask_token::String
    do_lower_case::Bool
end

function WordPieceTokenizer(vocab_file::String;
    do_lower_case=true,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]")
    vocab = load_vocab(vocab_file)
    ids_to_tokens = Dict(v => k for (k, v) in vocab)
    return WordPieceTokenizer(vocab, ids_to_tokens, unk_token, sep_token, pad_token, cls_token, mask_token, do_lower_case)
end

function load_vocab(vocab_file::String)
    vocab = Dict{String,Int}()
    open(vocab_file, "r") do f
        for (i, line) in enumerate(eachline(f))
            token = strip(line)
            if !isempty(token)
                vocab[token] = i
            end
        end
    end
    return vocab
end

function is_punctuation(char::Char)
    return ispunct(char) || (char in ['-', '_', '.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '"', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '/', '\\', '|'])
end

function basic_tokenize(text::String, do_lower_case::Bool)
    if do_lower_case
        text = lowercase(text)
    end

    tokens = String[]
    current_token = Char[]

    for char in text
        if isspace(char)
            if !isempty(current_token)
                push!(tokens, String(current_token))
                empty!(current_token)
            end
        elseif is_punctuation(char)
            if !isempty(current_token)
                push!(tokens, String(current_token))
                empty!(current_token)
            end
            push!(tokens, String([char]))
        else
            push!(current_token, char)
        end
    end

    if !isempty(current_token)
        push!(tokens, String(current_token))
    end

    return tokens
end

function wordpiece_tokenize(token::String, vocab::Dict{String,Int}, unk_token::String)
    chars = collect(token)
    len = length(chars)
    output_tokens = String[]
    start = 1

    while start <= len
        end_idx = len
        cur_substr = nothing
        found = false

        while start <= end_idx
            substr = String(chars[start:end_idx])
            if start > 1
                substr = "##" * substr
            end

            if haskey(vocab, substr)
                cur_substr = substr
                push!(output_tokens, cur_substr)
                start = end_idx + 1
                found = true
                break
            end
            end_idx -= 1
        end

        if !found
            return [unk_token]
        end
    end

    return output_tokens
end

function tokenize(tokenizer::WordPieceTokenizer, text::String)
    basic_tokens = basic_tokenize(text, tokenizer.do_lower_case)
    wordpiece_tokens = String[]

    for token in basic_tokens
        subtokens = wordpiece_tokenize(token, tokenizer.vocab, tokenizer.unk_token)
        append!(wordpiece_tokens, subtokens)
    end

    return wordpiece_tokens
end

function encode(tokenizer::WordPieceTokenizer, text::String; max_length=nothing, pad_to_max_length=false)
    tokens = tokenize(tokenizer, text)

    # Add special tokens
    tokens = [tokenizer.cls_token; tokens; tokenizer.sep_token]

    # Convert to IDs
    unk_id = get(tokenizer.vocab, tokenizer.unk_token, 0)
    ids = [get(tokenizer.vocab, t, unk_id) for t in tokens]

    if max_length !== nothing
        if length(ids) > max_length
            ids = ids[1:max_length-1]
            push!(ids, get(tokenizer.vocab, tokenizer.sep_token, 0))
        elseif pad_to_max_length
            pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
            while length(ids) < max_length
                push!(ids, pad_id)
            end
        end
    end

    return ids
end

"""
    encode_batch(tokenizer, texts; max_length=512, padding=:longest)

Encode multiple texts into a batch matrix with padding and attention mask.

# Arguments
- `texts::Vector{String}`: Input texts to encode
- `max_length::Int`: Maximum sequence length (default: 512)
- `padding::Symbol`: Padding strategy - `:longest` or `:max_length` (default: `:longest`)

# Returns
- `input_ids::Matrix{Int}`: Shape (seq_len, batch_size) - token IDs
- `attention_mask::Matrix{Float32}`: Shape (seq_len, batch_size) - 1.0 for real tokens, 0.0 for padding
"""
function encode_batch(tokenizer::WordPieceTokenizer, texts::Vector{String};
    max_length::Int=512, padding::Symbol=:longest)
    # Encode all texts
    all_ids = [encode(tokenizer, t) for t in texts]

    # Determine target length based on padding strategy
    max_actual_len = maximum(length.(all_ids))
    target_len = if padding == :longest
        min(max_length, max_actual_len)
    else
        max_length
    end

    # Create output matrices
    pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
    batch_size = length(texts)
    input_ids = fill(pad_id, target_len, batch_size)
    attention_mask = zeros(Float32, target_len, batch_size)

    for (i, ids) in enumerate(all_ids)
        len = min(length(ids), target_len)
        input_ids[1:len, i] = ids[1:len]
        attention_mask[1:len, i] .= 1.0f0
    end

    return input_ids, attention_mask
end

end
