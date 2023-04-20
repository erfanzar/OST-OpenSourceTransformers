from transformers import LlamaTokenizer, Trainer, LlamaForCausalLM, LlamaConfig
from modules import LGeMForCausalLM, LGeMConfig
import torch
from erutils import make2d
import transformers
from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType
import textwrap
from os import system, name
from sys import stdout
from IPython.display import clear_output

tokenizer = LlamaTokenizer.from_pretrained('erfanzar/LGeM-7B')


def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux
    else:
        _ = system('clear')


tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = LlamaForCausalLM.from_pretrained('erfanzar/LGeM-100M/checkpoint-37000')

if __name__ == "__main__":
    verify_text = lambda txt: '\n'.join([textwrap.fill(txt, width=120) for txt in txt.split('\n')])
    input_ids = tokenizer.encode(
        'User: Who are some students at Hogwarts in Hufflepuff house that were in the same year as Harry Potter? <|endoftext|>Assistant:',
        return_tensors='pt', add_special_tokens=False)

    while True:
        input_ids = model.generate(make2d(input_ids),
                                   max_new_tokens=1)
        # clear_output(True)
        stdout.flush()
        stdout.write(f'\r{verify_text(tokenizer.decode(input_ids[0]))}')

        if input_ids[0][-1] == tokenizer.eos_token_id:
            break
