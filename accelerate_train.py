import torch.nn
from transformers import AutoConfig, AutoModelForCausalLM
import accelerate
from utils.utils import count_model_parameters
from argparse import ArgumentParser
import torch
import os
from utils.timer import Timers
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from transformers import get_scheduler

LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))

AVAILABLE_OPTIMIZER = [
    'adamw_torch',
    'adamw_deepspeed',
    'adamw_32bit',
    'adamw_8bit',
    'lamb_32bit',
    'lamb_8bit',
]


def print_rank_0(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)


def get_argument_parser():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--cls_to_wrap', default='LtBlock', type=str,
                                 help='transformer layer class to warp for fully sharded data parallel')

    argument_parser.add_argument('--model_id', default='erfanzar/LT-500M', type=str,
                                 help='model id to save output and push to hub')

    argument_parser.add_argument('--tokenizer_id', default='erfanzar/LT-500M', type=str,
                                 help='tokenizer repo id to read from the specific tokenizer that you want '
                                      'to use on model')

    argument_parser.add_argument('--dataset', default='erfanzar/Base-Data', type=str,
                                 help='hugging face dataset to download or path to data.json file')

    argument_parser.add_argument('--dataset_field', default='prompt', type=str,
                                 help='the specific field ub dataset to look for and run tokenizer on that')
    argument_parser.add_argument('--max_length', default=768, type=int,
                                 help='train max sequence length')

    argument_parser.add_argument('--num_train_epochs', default=3, type=int,
                                 help='num train epochs')

    argument_parser.add_argument('--from_safetensors', action='store_true',
                                 help='load model from safetensors checkpoints instance of .bin .pt or .pth')

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
    argument_parser.add_argument('--resume_from_checkpoint', default=False, action='store_true',
                                 help='resume from the last trained checkpoint')

    argument_parser.add_argument('--optimizer', default='adamw_torch', type=str,
                                 help=f'Optimizer to train model available optimizers Are : {AVAILABLE_OPTIMIZER}')
    argument_parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                                 help='gradient accumulation steps')
    argument_parser.add_argument('--do_compile',
                                 help='compile the model')

    argument_parser.add_argument('--gradient_checkpointing', default=False, action='store_true',
                                 help='use gradient checkpointing or not its better to use cause make '
                                      'training better and lighter')
    args = argument_parser.parse_args()
    return args


def configure_optimizer(optimizer_type_, optimizer_kwargs_):
    if optimizer_type_ == 'adamw_deepspeed':
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(
            **optimizer_kwargs_
        )
    elif optimizer_type_ == 'adamw_torch':
        optimizer = torch.optim.AdamW(
            **optimizer_kwargs_
        )
    elif optimizer_type_ == 'adamw_32bit':
        from bitsandbytes.optim import AdamW32bit
        optimizer = AdamW32bit(
            **optimizer_kwargs_
        )
    elif optimizer_type_ == 'adamw_8bit':
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(
            **optimizer_kwargs_
        )
    elif optimizer_type_ == 'lamb_32bit':
        from bitsandbytes.optim import LAMB32bit
        optimizer = LAMB32bit(
            **optimizer_kwargs_
        )
    elif optimizer_type_ == 'lamb_8bit':
        from bitsandbytes.optim import LAMB8bit
        optimizer = LAMB8bit(
            **optimizer_kwargs_
        )
    else:
        raise ValueError('Invalid Optimizer')
    return optimizer


def main():
    args = get_argument_parser()
    accelerator = accelerate.Accelerator(
        logging_dir=f"{args.model_id}/logs",
        project_dir=f"{args.model_id}",
        gradient_accumulation_steps=args.gradient_accumulation_steps

    )
    log_t_path = Path('out\\performance_metrics')
    log_t_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=f'{log_t_path}')
    timers = Timers(
        use_wandb=False,
        tensorboard_writer=writer
    )

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config.hidden_size = 256
    config.num_hidden_layers = 2
    config.intermediate_size = 512
    config.num_attention_heads = 8
    num_training_steps = 1000
    timers('create model').start()
    with accelerate.init_empty_weights():
        model: torch.nn.Module = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    timers('create model').end()
    timers.log('create model')

    timers('config optimizer and scheduler').start()
    optimizer_kwargs = dict(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2
    )
    optimizer = configure_optimizer(optimizer_type_=args.optimizer, optimizer_kwargs_=optimizer_kwargs)
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        name=args.lr_scheduler_type,
        num_training_steps=num_training_steps
    )
    timers('config optimizer and scheduler').stop()
    timers.log('config optimizer and scheduler')

    # TODO


if __name__ == "__main__":
    main()
