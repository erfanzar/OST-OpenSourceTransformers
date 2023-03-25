import argparse
import logging

import torch.utils.data
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.train import train
from modules import LLMoUModel
from modules.datasets import DatasetLLMoU
from utils.utils import get_data, get_config_by_name

torch.manual_seed(42)

torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--accumulate', '--accumulate', type=int, default=4)
pars.add_argument('--out-path', '--out-path', type=str, default='out')
pars.add_argument('--model', '--model', type=str, default='LGeM-S')
pars.add_argument('--data-src', '--data-src', type=str, default='data/alpaca_data.json')
# HF-kilt_tasks//eli5
options = pars.parse_args()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARN)


def main(opt):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer_model/BASE')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    data = get_data(opt.data_src)[:850]
    conf = get_config_by_name(opt.model)
    # Replace with your own Dataset
    dataset = DatasetLLMoU(data=data, max_length=conf.max_sentence_length, tokenizer=tokenizer)
    train(model_type=opt.model,
          gradient_accumulation_steps=opt.accumulate,
          model_class=LLMoUModel,
          batch_size=opt.batch,
          dataset=dataset,
          weight=opt.weight,
          out_path=opt.out_path,
          )


if __name__ == "__main__":
    main(options)
