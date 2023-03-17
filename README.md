# PTT

researching for some in NLP are

PTT is a model Collection that includes some models that perform well and its open source and the weights are available
to

![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)

## Models

- PTT Project Contain currently 5 Models

### LLama 🧠

- First model is LLama (LLama is the same model as Meta (old Facebook) model but had some developments )

- it's Decoder Only
- built in Pytorch
- you can simply import model like

```python
from modules import LLamaModel
```

- and Training code is available at LLama-Train.py (check source)

```shell
python3 LLama-train.py
```

### LLMoU 🧠

- LLMoU is an NLP model fast and good enough to play around with

- it's Decoder Only
- and have configs start from LLMoU-S to LLMoU-LLX
- built in Pytorch
- you can simply import model like

```python
from modules import LLMoUModel
```

- and Training code is available at LLMoU-Train.py (check source)

```shell
python3 LLMoU-train.py
```

### LLmP 🧠

- LLmP is one of the best current models in this project that uses ALiBi, and it's kinda the best Model in the series

- it's Decoder Only
- and have configs start from LLmP-S to LLmP-LLX
- built in Pytorch
- you can simply import model like

```python
from modules import LLmP
```

- and Training code is available at LLmP-Train.py (check source)

```shell
python3 LLmP-train.py
```

### LLmPU 🧠

- LLmPU is Decoder Encoder (Transformer) and it's working perfectly fine

- it's Decoder Encoder
- and have configs start from LLmPU-S to LLmPU-LLX
- built in Pytorch and using transformers from huggingface
- you can simply import model like
- weight are Available for Pytorch

```python
# for simple training
from modules import LLmPUModel
# for use and generate [interface]
from modules import LLmPUForConditionalGeneration
```

- and Training code is available at LLmPU-Train.py (check source)

```shell
python3 LLmPU-train.py
```

### PGT 🧠

- PGT (Poetry Generated Transformers [funny name :) ]) is actually a nice model that can perform very nicely in
  multitask command and I recommend to train it with specific tasks and the weight will be available soon to use
  around (3.9 B)

- it's Decoder Only
- and have configs start from PGT-S to PGT-LLX
- built in Pytorch
- you can simply import model like

```python
from modules import PGT
```

- and Training code is available at PGT-Train.py (check source)

```shell
python3 PGT-train.py
```

## Charts 📊

#TODO charts have to be added

[//]: # (| Parameter   | Size  | number_of_embedded | chunk_size | head_size | number_of_head | number_of_layers    |)

[//]: # (|:------------|:------|:-------------------|------------|-----------|----------------|---------------------|)

[//]: # (| `32 M`      | 332M  | 384                | 128        | 64        | 8              | 18                  |)

## 🚀 About Me

Hi there 👋

I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## License

it's available at [MIT](https://choosealicense.com/licenses/mit/) license But feel Free to Use :) <3

## Contributing

Contributions are always welcome!

email at Erfanzare82@yahoo.com

## Used By

This project is used by the following companies:

- You Can Be First One Here :)

## Author

- hello i am [@erfanzar](https://www.github.com/erfanzar)

