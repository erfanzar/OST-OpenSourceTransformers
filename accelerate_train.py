import math

import torch.nn
from transformers import AutoConfig, AutoModelForCausalLM
import accelerate
from utils.utils import count_model_parameters

from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch
from tqdm.auto import tqdm
import os
from utils.timer import Timers
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from transformers import get_scheduler, AutoTokenizer
from datasets import load_dataset

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

    argument_parser.add_argument('--from_config', default='none',
                                 help='Build model from a config')

    argument_parser.add_argument('--gradient_checkpointing', default=False, action='store_true',
                                 help='use gradient checkpointing or not its better to use cause make '
                                      'training better and lighter')

    argument_parser.add_argument('--trust_remote_code', action='store_true',
                                 help='trust_remote_code to load model code from huggingface Repo')
    args = argument_parser.parse_args()
    return args


def configure_dataset(accelerator, batch_size: int, dataset_, dataset_field_, tokenizer, max_length_: int,
                      is_pre_training=False):
    data = load_dataset(dataset_)
    if is_pre_training:
        raise NotImplementedError
    with accelerator.main_process_first():
        data = data.map(lambda x_: tokenizer(x_[dataset_field_], max_length=max_length_, padding='max_length'),
                        desc=f'MAPPING DATASET | WORLD SIZE : {WORLD_SIZE}  ', num_proc=WORLD_SIZE,
                        with_rank=True)
    return DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True), data.num_rows


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
    timers('loading tokenizer and config').start()
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config.hidden_size = 256
    config.num_hidden_layers = 2
    config.intermediate_size = 512
    config.num_attention_heads = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    timers('loading tokenizer and config').stop()
    timers.log('loading tokenizer and config')
    timers('loading data').start()
    save_strategy = args.save_strategy
    dataloader_train, num_rows = configure_dataset(accelerator=accelerator, dataset_=args.dataset,
                                                   dataset_field_=args.dataset_field,
                                                   max_length_=args.max_length, batch_size=args.per_device_batch_size,
                                                   tokenizer=tokenizer)
    timers('loading data').stop()
    timers.log('loading data')

    # Calculate Training Steps and configs
    num_training_steps = num_rows * args.num_train_epochs
    num_training_steps_w_batch = math.ceil(num_training_steps / args.per_device_batch_size)
    num_training_steps_w_batch_gradient_ac = math.ceil(num_training_steps_w_batch / args.gradient_accumulation_steps)

    print_rank_0(f'Num Training Examples : {num_training_steps}')
    print_rank_0(f'Num Training Examples / Batched : {num_training_steps_w_batch}')
    print_rank_0(f'Num Training Examples / Batched / GradientAccumulation  : {num_training_steps_w_batch_gradient_ac}')

    timers('create model').start()
    if args.from_config == 'none':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_config(args.from_config, trust_remote_code=args.trust_remote_code)
    print_rank_0(
        f'Model Contain {count_model_parameters(model, 1e9)} BILLION Parameters'
    )
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
        num_training_steps=num_training_steps_w_batch_gradient_ac
    )
    timers('config optimizer and scheduler').stop()
    timers.log('config optimizer and scheduler')
    timers('training').start()
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    train_data = accelerator.prepare_data_loader(data_loader=dataloader_train)
    scheduler = accelerator.prepare_scheduler(scheduler=scheduler)
    for epoch in tqdm(range(args.num_train_epochs)):

        for step, batch in tqdm(enumerate(dataloader_train)):
            with accelerator.accumulate(model):
                batch.to(accelerator.device)
                pred = model(**batch, return_dict=True)
                loss = pred.loss
                accelerator.backward(loss=loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % args.logging_step == 0:
                    accelerator.print(
                        f'Loss : {loss} | Learning Rate : {scheduler.get_lr()} | Epoch {epoch} / {args.num_train_epochs}'
                    )
                if save_strategy == 'steps' and (step + 1) % args.save_steps == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(
                        {
                            'model': unwrapped_model.state_dict(),
                            'optimizer': optimizer.optimizer.state_dict(),
                            'args': {args}
                        }, args.model_id + '/pytorch_model.bin'
                    )
        if save_strategy == 'epoch':
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.optimizer.state_dict(),
                    'args': {args}
                }, args.model_id + '/pytorch_model.bin'
            )
    timers('training').stop()
    timers.log('training')


if __name__ == "__main__":
    main()
