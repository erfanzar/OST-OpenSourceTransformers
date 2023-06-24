# OST

some researchs in `NLP`

OST Collection: An AI-powered suite of models that predict the next word matches with remarkable accuracy (Text
Generative Models). OST Collection is based on a novel approach to work as a full and intelligent NLP Model.

## Trained Available Models 

| Model       Link                                                                 | Max Sentence Length | Parameters |
|:---------------------------------------------------------------------------------|---------------------|------------|
| [Mpt-7B-Assistant(Dragon) 🚀 ](https://huggingface.co/erfanzar/Mpt-7B-Assistant) | 5144                | 7B         | 
| [LGeM-13B-MT 🚀 ](https://huggingface.co/erfanzar/LGeM-13B-MT)                   | 2048                | 13B        | 
| [LGeM-7B-MT 🚀 ](https://huggingface.co/erfanzar/LGeM-7B-MT)                     | 2048                | 7B         | 
| [LGeM-7B-C 🚀 ](https://huggingface.co/erfanzar/LGeM-7B-C)                       | 2048                | 7B         | 
| [GT-J-6B 🚀 ](https://huggingface.co/erfanzar/GT-J)                              | 2048                | 6B         |    
| [LGeM-3.5B 🚀 ](https://huggingface.co/erfanzar/LGeM-3B5)                        | 2048                | 3.5B       |      
| [LGeM-1B 🚀 ](https://huggingface.co/erfanzar/LGeM-1B)                           | 1024                | 1B         | 
| [LGeM-7B 🚀 ](https://huggingface.co/erfanzar/LGeM-7B)                           | 2048                | 7B         | 
| [PGT-1B 🚀 ](https://huggingface.co/erfanzar/PGT-1B)                             | 1280                | 1B         |

# TODO

- Create a Tutorial on How to train or finetune your own LLM up to 8Billion parameters for free!

  
## Train or Finetune 

you have many options to choose which code to choose for train the models but we recommend using train.py that you can use fsdp and deepspeed 



DeepSpeed Example

```shell
deepspeed --no_python --master_addr=4008 --num_gpus=<number_of_your_gpus_here> train.py \
--use_deepspeed \
--dataset <your dataset> \
--dataset_field <field in dataset that tokenizer tokeniz > \
--max_length=<your_max_length> \
--auto_batch \
--save_safetensors \
--model_id='trainer' \
--no_resume_from_checkpoint \
--cls_to_wrap=<YourModelBlock> \
--logging_step=10 \
--report_to='wandb' \
--save_total_limit=2 \
--no_do_eval \
--lr_scheduler_type='cosine'
```

FSDP Example


```shell
torchrun --nproc-per-node=<number_of_your_gpus_here> --master-port=4008 --standalone train.py \
--use_fsdp \
--dataset <your dataset> \
--dataset_field <field in dataset that tokenizer tokeniz > \
--max_length=<your_max_length> \
--auto_batch \
--save_safetensors\
--model_id='trainer' \
--no_resume_from_checkpoint\
--cls_to_wrap=<YourModelBlock> \
--logging_step=10 \
--report_to='wandb' \
--save_total_limit=2 \
--no_do_eval \
--lr_scheduler_type='cosine'
```



### LT (LucidTransformers)-Models

- upcoming soon
- LLM
- uses ALIBI as positionnal embeddings significantly outperforms other embeddings for zero-shot generalization.
- flash attention
- 1B , 3B ,7B ,12B 50B
- context length 9K

### LGeM 🚀

- what is LGeM , LGeM is a CausalLM Model that trained on self instruct data (Alpaca data) and for initilization of the
  first train of main model (weight are available) I used pre weights from Alpaca LoRA (open source)

- it's Decoder Only
- built in Pytorch
- you can simply import model like

```python
from modules import LGeMForCausalLM
```

- and Training code is available at LGeM-Train.py (check source)
- training parameters
-
    - learning rate 1e-4
-
    - AdamW (weight decay 1e-2)
-
    - batch 2
-
    - A 100 80GB used for training (4 X)

```shell
python3 LGeM-train.py
```

#### Available at [Huggingface](https://huggingface.co/erfanzar/LGeM-7B)

### LLama 🚀

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

### LLMoU 🚀

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

### LLmP 🚀

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

### LLmPU 🚀

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

### PGT 🚀

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

| Model      | Hidden size | number of Layers | number of Heads | Max Sentence Length | Parameters  |
|:-----------|:------------|:-----------------|-----------------|---------------------|-------------|
| PGT-S      | 768         | 10               | 12              | 256                 | 148.62 M    | 
| PGT-M      | 1024        | 18               | 12              | 512                 | > 15 B      | 
| PGT-X      | 1536        | 28               | 16              | 512                 | 947.30 M    | 
| PGT-LX     | 2048        | 34               | 32              | 768                 | 1,917.49 B  | 
| PGT-LXX    | 4096        | 64               | 32              | 2000                | 13,297.54 B | 
| LLama      | 4096        | 18               | 16              | 256                 | 5,243.83 B  | 
| LLmP-S     | 768         | 10               | 8               | ALiBi               | 148.82 M    | 
| LLmP-ML    | 1024        | 18               | 16              | ALiBi               | > 15 B      | 
| LLmP       | 1536        | 24               | 16              | ALiBi               | 834.00 M    | 
| LLmP-X     | 1792        | 36               | 16              | ALiBi               | 1,567.58 B  | 
| LLmP-L     | 2048        | 32               | 32              | ALiBi               | 1,816.68 B  | 
| LLmP-LX    | 4096        | 48               | 32              | ALiBi               | > 15 B      | 
| LLMoU-S    | 768         | 10               | 8               | 512                 | 148.14 M    | 
| LLMoU-ML   | 1024        | 18               | 16              | 512                 | 329.71 M    | 
| LLMoU      | 1536        | 26               | 16              | 256                 | 891.03 M    | 
| LLMoU-X    | 2048        | 34               | 32              | 256                 | 1,918.02 B  | 
| LLMoU-L    | 2048        | 48               | 32              | 1024                | 2,622.98 B  | 
| LLMoU-LX   | 2048        | 52               | 32              | 2048                | > 15 B      | 
| LLmPU-base | 1792        | 8                | 12              | 512                 | 598.64 M    | 
| LLmPU-S    | 1024        | 6                | 12              | 256                 | 225.68 M    | 
| LLmPU-L    | 1792        | 10               | 12              | 768                 | 758.30 M    | 
| LLmPU-LX   | 2048        | 14               | 12              | 768                 | 1,791.52 B  | 

## 🚀 About Me

Hi there 👋

I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## Contributing

Contributions are always welcome!

email at Erfanzare82@yahoo.com

## Used By

This project is used by the following companies:

- You Can Be First One Here :)

## Author

- hello i am [@erfanzar](https://www.github.com/erfanzar)

## Reference & Papers used

[Hello, It's GPT-2 -- How Can I Help You? Towards the Use of Pretrained Language Models for Task-Oriented Dialogue Systems](https://arxiv.org/abs/1907.05774)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[ALiBi : Towards Accurate and Robust
Identification of Backdoor Attacks
in Federated Learning](https://arxiv.org/pdf/2202.04311.pdf)

[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
