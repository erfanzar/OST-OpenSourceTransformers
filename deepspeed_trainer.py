import time

import deepspeed

from modules import LGeMConfig, LGeMForCausalLM
import os
import multiprocessing as mp
from transformers import HfArgumentParser
import argparse


def get_argument_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_id', default='erfanzar/PaLM', type=str,
                                 help='model id to save output and push to hub')

    argument_parser.add_argument('--tokenizer_id', default='erfanzar/LT-500M', type=str,
                                 help='tokenizer repo id to read from the specific tokenizer that you want '
                                      'to use on model')

    argument_parser.add_argument('--dataset', default='erfanzar/Base-Data', type=str,
                                 help='hugging face dataset to download or path to data.json file')
    argument_parser.add_argument('--dont_tokenizer', action='store_true',
                                 help='for pretraining')
    argument_parser.add_argument('--dataset_field', default='prompt', type=str,
                                 help='the specific field ub dataset to look for and run tokenizer on that')
    argument_parser.add_argument('--max_length', default=4096, type=int,
                                 help='train max sequence length')

    argument_parser.add_argument('--num_train_epochs', default=3, type=int,
                                 help='num train epochs')

    argument_parser.add_argument('--per_device_batch_size', default=8, type=int,
                                 help='pre device batch size')
    argument_parser.add_argument('--learning_rate', default=1e-4, type=float,
                                 help='learning rate for the optimizer')
    argument_parser.add_argument('--lr_scheduler_type', default='cosine', type=str,
                                 help='learning rate scheduler type type for optimizer')

    argument_parser.add_argument('--logging_step', default=15, type=int,
                                 help='logging steps default to 15')

    argument_parser.add_argument('--save_steps', default=1500, type=int,
                                 help='steps to save model in training')

    argument_parser.add_argument('--save_strategy', default='epoch', type=str,
                                 help='save strategy [epoch or steps]')
    argument_parser.add_argument('--save_total_limit', default=1, type=int,
                                 help='total limit of saving model')

    argument_parser.add_argument('--do_compile',
                                 help='compile the model')

    argument_parser.add_argument('--from_config', default='none',
                                 help='Build model from a config')

    argument_parser.add_argument('--gradient_checkpointing', default=False, action='store_true',
                                 help='use gradient checkpointing or not its better to use cause make '
                                      'training better and lighter')

    argument_parser.add_argument('--trust_remote_code', action='store_true',
                                 help='trust_remote_code to load model code from huggingface Repo')
    args = argument_parser.parse_args()
    return args


def train():
    ...
    # TODO : CREATE DEEPSPEED TRAINER


def run():
    ...
    # TODO : CREATE DEEPSPEED TRAINER


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
