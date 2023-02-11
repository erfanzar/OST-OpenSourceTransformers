import os
import typing

import torch
from torch import nn as nn
from utils.utils import get_config_by_name, DatasetQA
from modules.models import PGT


def load_pgt(model_path: typing.Union[str, os.PathLike], model_name: str):
    dataset = DatasetQA()

    config = get_config_by_name(model_name, dataset.tokenizer.vocab_size)
    ckpt = torch.load(model_path, map_location=config.device)
    print(*(k for k, _ in ckpt.items()))
    model = PGT(config)
    model.load_state_dict(ckpt['model'])
    x = dataset.encode('hello how are you ?', True)
    input = x['input_ids']
    mask = x['attention_mask']
    pred = model.forward(input, mask)
    pred = pred.view(-1, pred.size(-1))

    pred = torch.nn.functional.softmax(pred, dim=-1)
    next_index = torch.multinomial(pred, 1).view(1, -1)
    print(dataset.decode(next_index))


if __name__ == "__main__":
    load_pgt('model.pt', 'PGT-ss')
