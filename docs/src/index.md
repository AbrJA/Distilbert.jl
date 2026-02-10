# Distilbert.jl

A pure Julia implementation of the DistilBERT model.

## API Reference

```@contents
```

### Model & Config

```@docs
DistilBertConfig
DistilBertModel
load_model
```

### Tokenizer

```@docs
WordPieceTokenizer
tokenize
load_vocab
encode
encode_pair
encode_batch
```

### Inference & Embeddings

```@docs
inference
embed
```

### Task-Specific Heads

```@docs
DistilBertForSequenceClassification
DistilBertForTokenClassification
DistilBertForQuestionAnswering
```

### Pooling

```@docs
cls_pooling
mean_pooling
max_pooling
```
