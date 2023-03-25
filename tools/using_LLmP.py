import os

os.chdir('/'.join(os.getcwd().split('\\')[:-1]))
import argparse

import torch.utils.data
from erutils.loggers import fprint
from transformers import AutoTokenizer

from modules.datasets import DatasetLLmP, Tokens
from modules.models import LLmP
from utils.utils import get_config_by_name, count_model_parameters, device_info

pars = argparse.ArgumentParser()
pars.add_argument('--model', '--model', type=str, default='LLmP-ML')
pars.add_argument('--agent-name', '--agent-name', type=str, default='<LLmP> :')
pars.add_argument('--tokenizer', '--tokenizer', type=str, default='tokenizer_model/LLmP-C')
opt = pars.parse_args()


def _main(options):
    device_info()

    config = get_config_by_name(options.model)
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
    dataset = DatasetLLmP(data=[], tokenizer=tokenizer)
    config.vocab_size = dataset.tokenizer.vocab_size
    config.vocab_size += 5
    # config.device = 'cpu'
    fprint('Loading Model ...')
    model: LLmP = LLmP(config=config).to('cpu')
    loaded = torch.load(f'{options.model}-model.pt', 'cpu')
    model.load_state_dict(loaded['model'])
    del loaded
    model = model.to(config.device)
    fprint(f'Model Loaded With {count_model_parameters(model)} Million Parameters')

    print('🧠Let Have Conversation Dude')

    model.eval()
    while True:
        income = input('>>> ')
        text = tokenizer.encode(Tokens.sos + income + options.agent_name, return_tensors='pt').to(config.device)

        for v in model.generate(text, max_gen_len=240, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id):
            print(f'{tokenizer.decode(v[0], skip_special_tokens=True)}', end='')
        print()


if __name__ == "__main__":
    _main(opt)
