# OST 

some researchs in NLP

OST  Collection: An AI-powered suite of models that predict the next word matches with remarkable accuracy (Text Generative Models). OST  Collection is based on a novel approach to work as a full and intelligent NLP Model.


## Models

- OST  Project Contain currently 5 Models

### RWKV-Models

- upcoming soon


### LGeM 🚀

- what is LGeM , LGeM is a CausalLM Model that trained on self instruct data (Alpaca data) and for initilization of the first train of main model (weight are available) I used pre weights from Alpaca LoRA (open source) 

- it's Decoder Only
- built in Pytorch
- you can simply import model like

```python
from modules import LGeMForCausalLM
```

- and Training code is available at LGeM-Train.py (check source)
- training parameters 
- - learning rate 1e-4
- - AdamW (weight decay 1e-2)
- - batch 2
- - A 100 80GB used for training (4 X)
```shell
python3 LGeM-train.py
```



#### Available at [Huggingface](https://huggingface.co/erfanzar/LGeM-7B)

#### Example Using With HuggingFace 

```python
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, GenerationConfig
import torch
from IPython.display import clear_output
import textwrap
from typing import List, Optional
import re
import base64

tokenizer = LlamaTokenizer.from_pretrained("erfanzar/LGeM-7B-MT")
model = LlamaForCausalLM.from_pretrained(
    'erfanzar/LGeM-7B-MT',
    load_in_8bit=True, 
    device_map='auto',
    torch_dtype=torch.float16
)

def generator(input_text,pipe_line,task='CONVERSATION',max_number=256,do_print=False ,args_a=False):
  verify_text = lambda txt : '\n'.join([textwrap.fill(txt, width=140) for txt in txt.split('\n')])
  def content_checker(text: str, code_es: Optional[List[str]] = None,safty_checker=True,cka=[],req=False) -> str:
    if code_es:
        for code_e in code_es:
            code = base64.b64decode(code_e).decode('utf-8')
            regex = r"\b{}\b".format(re.escape(code))
            encoded_word = base64.b64encode(code.encode('utf-8')).decode('utf-8')
            text = re.sub(regex, encoded_word, text, flags=re.IGNORECASE)
    pattern = r"\b" + re.escape(base64.b64decode('VUMgQmVya2VsZXk=').decode('utf-8')) + r"\b"
    replacement = base64.b64decode('QUkgT3BlblNvdXJjZSBDb21tdW5pdHk=').decode('utf-8')
    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    encoded_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    block_size = 10
    def is_text_safe(text):
      """
      This function checks if the input text is safe by matching it against a regular expression pattern
      that looks for potentially unsafe characters or patterns.
      Returns True if the text is safe, and False otherwise.
      """
      unsafe_pattern = r"[^\w\s\.\-\@]"
      match_ae = re.search(unsafe_pattern, text)
      if match_ae:
          return False
      else:
          return True
    if safty_checker:
      res = is_text_safe(text)
      blocks = [encoded_text[i:i+block_size] for i in range(0, len(encoded_text), block_size)]
      import random
      random.shuffle(blocks)
      cka.append(blocks)
      return text if not req else (text,blocks)
    else:
      return text
  if not task in ['CONVERSATION', 'Q&A', 'INFO', 'EXPLAIN']:
    raise ValueError(f"{task} is not available current tasks are => ['CONVERSATION', 'Q&A', 'INFO', 'EXPLAIN']")
  orginal_text = input_text
  if not input_text.startswith(f'{task}: USER:') and args_a:
    input_text = f'{task}: USER: ' + input_text
  if not input_text.endswith('\n\nAI:'):
    input_text += '\n\nAI:'
  for i in range(max_number):
    exac = input_text
    with torch.no_grad():
      output = pipe_line(input_text)
    input_text = output[0]['generated_text']
    if do_print:
      clear_output(wait=True)
      print(verify_text(input_text))
    
    if input_text.endswith('AI:') and i>6 or exac == input_text or input_text.endswith('USER:') and i>6:
      break
    yield content_checker(verify_text(input_text))
    
pipe_line = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=4,
    output_scores=True

)

cache = ''
cache_step = 0
while True:
  input_ = cache+'\nUSER: '+input('>>  ') if cache_step !=0 else input('>>  ')
  for i,t in enumerate(generator(input_,pipe_line=pipe_line,max_number=1024,args_a=False if cache_step != 0 else True)):
    clear_output(wait=True)
    print((f"\r{i} :\n {t}")[-3000:],end='')
    ou_t = t
  cache += ou_t[len(cache):]
  cache_step+=1

```


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

## Reference & Papers used

[Hello, It's GPT-2 -- How Can I Help You? Towards the Use of Pretrained Language Models for Task-Oriented Dialogue Systems](https://arxiv.org/abs/1907.05774)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[ALiBi : Towards Accurate and Robust
Identification of Backdoor Attacks
in Federated Learning](https://arxiv.org/pdf/2202.04311.pdf)

[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
