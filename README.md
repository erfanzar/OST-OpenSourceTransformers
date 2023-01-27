# PTT

some researchs to make an NLP Ai model that kinda work like GPT but just it's just a baby yet (Using some custom
Attention and MultiHeadAttentions and CasualAttention Built form orginal paper with adding a bit of research to it)

## Training

To train PTT project run

```bash
  python3 engine.py --config-path <config/config.yaml>
```

## API Reference

#### PTTCasuialAttention

```python
  PTTCasuialAttention(
    vocab_size: int,
    number_of_layers: int,
    number_of_embedded: int,
    head_size: int,
    number_of_head: int,
    chunk_size: int
    ) -> PTT_MODEL:
...
```

| Parameter   | Size    | number_of_embedded | chunk_size | head_size | number_of_head | number_of_layers |
|:------------|:--------|:-------------------|------------|-----------|----------------|------------------|
| `120 M ~`   | 332 M   | 728                | 256        | 128       | 14             | 20               |

#### PTTMultiHeadAttention

```python
  PTTMultiHeadAttention(
    vocab_size: int,
    number_of_layers: int,
    number_of_embedded: int,
    head_size: int,
    number_of_head: int,
    chunk_size: int
    ) -> PTT_MODEL:
...
```

| Parameter | Size     | number_of_embedded | chunk_size | head_size | number_of_head | number_of_layers |
|:----------|:---------|:-------------------|------------|-----------|----------------|------------------|
| `32 M`    | 332 M    | 384                | 128        | 64        | 8              | 18               |

## 🚀 About Me

Hi there 👋
I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.
