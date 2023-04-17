import os
import sys

import torch

sys.path.append('/'.join(os.getcwd().split('\\')[:-1]))
import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--model-type', '--model-type', type=str, default='LGeM-SM')
pars.add_argument('--tokenizer', '--tokenizer', type=str, default='erfanzar/LGeM-7B')
opt = pars.parse_args()
os.environ['USE_JIT'] = '1'
from modules import LGeMForCausalLM
import accelerate
from utils.utils import get_config_by_name, count_model_parameters, prompt_to_instruction
from transformers import AutoTokenizer, PreTrainedTokenizer


def _main(options):
    accelerator = accelerate.Accelerator()

    data = torch.load(r'D:\OST-OpenSourceTransformers\out\LGeM-SM\weights\LGeM-SM-model.pt')
    print(*(k for k, v in data.items()))
    conf = data['configuration']
    model = LGeMForCausalLM(conf)
    model.load_state_dict(data['model'])
    model = model.to('cuda:0')
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        r'D:\OST-OpenSourceTransformers\erfanzar/LGeM-7B')

    prompt = prompt_to_instruction('Give three tips for staying healthy.')
    encoded = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        pred = model(encoded.to('cuda:0'))
        print(pred)


if __name__ == "__main__":
    _main(opt)
