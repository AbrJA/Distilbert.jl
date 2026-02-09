using Test
using Distilbert
using Flux

@testset "Distilbert.jl" begin
    @testset "Config" begin
        config = DistilBertConfig()
        @test config.dim == 768
        @test config.n_layers == 6
        @test config.n_heads == 12
        @test config.vocab_size == 30522
    end

    @testset "Forward Pass Shape" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)

        batch_size = 2
        seq_len = 10
        input_ids = rand(1:100, seq_len, batch_size)

        output = model(input_ids)

        # Expected output shape: (dim, seq_len, batch_size)
        @test size(output) == (64, seq_len, batch_size)
    end

    @testset "Embeddings Layer" begin
        config = DistilBertConfig(dim=64, n_heads=4, vocab_size=100)
        emb = Distilbert.Embeddings(config)
        x = rand(1:100, 10, 2)
        y = emb(x)
        @test size(y) == (64, 10, 2)
    end

    @testset "Attention Layer" begin
        config = DistilBertConfig(dim=64, n_heads=4)
        attn = Distilbert.MultiHeadSelfAttention(config)
        x = randn(Float32, 64, 10, 2)  # (dim, seq_len, batch_size)
        y = attn(x)
        @test size(y) == (64, 10, 2)
    end

    @testset "Tokenizer" begin
        # Create a minimal vocab for testing
        vocab_content = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n##ing\ntest\n"
        vocab_file = tempname()
        write(vocab_file, vocab_content)

        tokenizer = WordPieceTokenizer(vocab_file; do_lower_case=true)

        # Test basic tokenization
        tokens = tokenize(tokenizer, "hello")
        @test "hello" in tokens

        # Test encoding adds special tokens
        ids = encode(tokenizer, "hello")
        @test length(ids) >= 3  # At least [CLS], hello, [SEP]
        @test ids[1] == tokenizer.vocab["[CLS]"]
        @test ids[end] == tokenizer.vocab["[SEP]"]

        rm(vocab_file)
    end
end
