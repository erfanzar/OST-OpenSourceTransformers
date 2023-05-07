import time

import deepspeed

from modules import LGeMConfig, LGeMForCausalLM
import os
import multiprocessing as mp
from transformers import HfArgumentParser
import argparse


def add_argument():
    parser = argparse.ArgumentParser(description='OST DeepSpeed')
    parser.add_argument('--with_cuda', default=False, action='store_true')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=30, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    print(args)
    return args


def train():
    ...


def run():
    ...


def main():
    deepspeed.init_distributed()
    args = add_argument()
    config = LGeMConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
    )
    model = LGeMForCausalLM(config=config)
    model_p = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, _ = deepspeed.initialize(args=args, model_parameters=model_p, model=model)


# pipeline('text_generation',)
if __name__ == '__main__':
    main()

