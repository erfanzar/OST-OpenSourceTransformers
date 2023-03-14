import json
import logging
import os
from typing import Union, Tuple

import torch
from transformers import AutoTokenizer, BasicTokenizer

from modules.modeling_LLmPU import LLmPUForConditionalGeneration, LLmPUConfig

logger = logging.getLogger(__name__)


def load_llmpu(config_path: Union[os.PathLike, str], model_ckpt: Union[os.PathLike, str],
               tokenizer_path: Union[os.PathLike, str]) -> Tuple[torch.nn.Module, BasicTokenizer]:
    logger.info(f'Loading Configs from {config_path}')
    config_kwargs = json.load(open(config_path, 'r'))
    logger.info(f'Configs Loaded Successfully')
    config = LLmPUConfig(**config_kwargs)
    logger.info(f'Loading Tokenizer from {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f'Tokenizer Loaded Successfully')
    logger.info(f'Creating Model')
    model = LLmPUForConditionalGeneration(config=config)
    logger.info(f'Model Created Successfully')
    logger.info(f'Loading Model from {model_ckpt}')
    ckpt = torch.load(model_ckpt)
    # logger.info(f'Available Options on last save are : {[k for k, v in ckpt.items()]}')
    model.load_state_dict(ckpt)
    logger.info(f'Model Loaded Successfully')

    return model, tokenizer
