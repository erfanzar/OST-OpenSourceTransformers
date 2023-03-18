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

| Model        | Hidden size | number of Layers | number of Heads | Max Sentence Length | Parameters  |
|:-------------|:------------|:-----------------|-----------------|---------------------|-------------|
| PGT-S        | 768         | 10               | 12              | 256                 | 148.62 M    | 
| PGT-M        | 1024        | 18               | 12              | 512                 | > 15 B      | 
| PGT-X        | 1536        | 28               | 16              |  512                |   947.30 M  | 
| PGT-LX       | 2048        | 34               | 32              | 768                 | 1,917.49 B  | 
| PGT-LXX      | 4096        | 64               | 32              | 2000                | 13,297.54 B | 
| LLama        | 4096        | 18               | 16              | 256                 | 5,243.83 B  | 
| LLmP-S       | 768         | 10               | 8               | ALiBi               | 148.82 M    | 
| LLmP-ML      | 1024        | 18               | 16              | ALiBi               | > 15 B      | 
| LLmP         | 1536        | 24               | 16              | ALiBi               | 834.00 M    | 
| LLmP-X       | 1792        | 36               | 16              | ALiBi               | 1,567.58 B  | 
| LLmP-L       | 2048        | 32               | 32              | ALiBi               | 1,816.68 B  | 
| LLmP-LX      | 4096        | 48               | 32              | ALiBi               | > 15 B      | 
| LLMoU-S      | 768         | 10               | 8               | 512                 | 148.14 M    | 
| LLMoU-ML     | 1024        | 18               | 16              | 512                 | 329.71 M    | 
| LLMoU        | 1536        | 26               | 16              | 256                 | 891.03 M    | 
| LLMoU-X      | 2048        | 34               | 32              | 256                 | 1,918.02 B  | 
| LLMoU-L      | 2048        | 48               | 32              | 1024                | 2,622.98 B  | 
| LLMoU-LX     | 2048        | 52               | 32              | 2048                | > 15 B      | 
| LLmPU-base   | 1792        | 8                | 12              | 512                 | 598.64 M    | 
| LLmPU-S      | 1024        | 6                | 12              | 256                 | 225.68 M    | 
| LLmPU-L      | 1792        | 10               | 12              | 768                 | 758.30 M    | 
| LLmPU-LX     | 2048        | 14               | 12              | 768                 | 1,791.52 B  | 



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

