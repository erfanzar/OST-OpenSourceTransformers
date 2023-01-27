
# PTT
some researchs to make an NLP Ai model that kinda work like GPT but just it's just a baby yet (Using some custom Attention and MultiHeadAttentions and CasualAttention Built form orginal paper with adding a bit of research to it)


![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Training

To train PTT project run

```bash
  python3 engine.py --config-path <config/config.yaml>
```

## Model Reference (PTTMultiHeadAttention)

| Model                     | Param  |                           
|---------------------------|--------|
| PTTMultiHeadAttention S   | `12 M` |
| PTTMultiHeadAttention N   | `32 M` |
| PTTMultiHeadAttention M   | `~ M`  |
| PTTMultiHeadAttention L   | `~ M`  |


## Model Reference (PTTCasualHeadAttention)

| Model                      | Param   |                           
|----------------------------|---------|
| PTTCasualHeadAttention S   | `~ M`   |
| PTTCasualHeadAttention N   | `~ M`   |
| PTTCasualHeadAttention M   | `~ M`   |
| PTTCasualHeadAttention L   | `~ M`   |

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


| Parameter       | Size   | number_of_embedded | chunk_size | head_size | number_of_head | number_of_layers |
|:----------------|:-------|:-------------------|------------|-----------|----------------|------------------|
| `120 M ~`       | 332 M  | 728                | 256        | 128       | 14             | 20               |

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

| Parameter   | Size  | number_of_embedded | chunk_size | head_size | number_of_head | number_of_layers    |
|:------------|:------|:-------------------|------------|-----------|----------------|---------------------|
| `32 M`      | 332M  | 384                | 128        | 64        | 8              | 18                  |




## 🚀 About Me
Hi there 👋
I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Used By

This project is used by the following companies:

- You Can Be First One Here :)



## Author

- [@erfanzar](https://www.github.com/erfanzar)

