import argparse
import logging
import math
import typing
from typing import Optional, Union

import erutils
import torch.utils.data
from datasets import load_dataset
from erutils.loggers import fprint
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

from modules.dataset import DatasetLLmP, Tokens
from modules.models import LLmP, LLmPConfig
from utils.utils import make2d, save_checkpoints, get_config_by_name, device_info

# logging.basicConfig(level=logging.DEBUG)
if __name__ == "__main__":
    parameters: LLmPConfig = get_config_by_name("LLmP")
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=Tokens.eos,
                                                             pad_token=Tokens.pad, sos_token=Tokens.sos)

    erutils.loggers.show_hyper_parameters(parameters)
    parameters.device = 'cpu'
    parameters.vocab_size = tokenizer.vocab_size + 1
    model: LLmP = LLmP(config=parameters).to(parameters.device)
    fpa = model.generate(tokenizer('hello how are you', return_tensors='pt').input_ids.to(parameters.device))

    print(fpa)
