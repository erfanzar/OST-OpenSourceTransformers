import math

import erutils
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,

)
import time
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    wrap,
    enable_wrap,
    transformer_auto_wrap_policy
)
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
import functools
import torch
from typing import Type, Tuple, Any
import torch.distributed as dist
from datetime import datetime
import torch.multiprocessing as mp
from pathlib import Path
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, logging, PreTrainedTokenizer, \
    HfArgumentParser
from dataclasses import field, dataclass
from modules.modeling_LGeM import LGeMBlock
from utils.utils import make2d, count_model_parameters

logging.set_verbosity_warning()
logger = logging.get_logger(__name__)
logger.setLevel('WARNING')

WARP_LAYER = LGeMBlock


@dataclass
class Arguments:
    model_id: str = field(default='facebook/opt-125m', init=True)
    scheduler: str = field(default='cosine',
                           metadata={'help': 'scheduler for optimizer options are ["cosine","constant","linear"]'})
    max_sequence_length: int = field(default=768, metadata={
        'help': 'max sequence length for train model'
    })
    dataset_name: str = field(default='erfanzar/AV30', metadata={
        'help': 'dataset name to train model on (any dataset from huggingface)'
    })
    dataset_field: str = field(default='LGeM', metadata={
        'help': 'a key in dataset["train"] to run tokenizer on '
    })
    use_fsdp: bool = field(default=True, metadata={
        'help': 'use Fully Sharded Data Parallel or not '
    })
    use_cpu_offload: bool = field(default=True, metadata={
        'help': 'use cpu offload for model'
    })
    batch_size: int = field(default=8, metadata={
        'help': 'batch size for train model default set to 8'
    })
    learning_rate: float = field(default=2e-4, metadata={
        'help': 'learning rate for model optimizers '
    })
    epochs: int = field(default=5)
    track_memory: bool = field(default=True, metadata={
        'help': 'track memory during training and validation'
    })
    num_processes: int = field(default=1, metadata={
        'help': 'number of processes run at the time set this to you machine number of gpus'
    })
    transformer_cls_to_wrap: Any = field(default=WARP_LAYER, metadata={
        'help': 'model Layer to be wraped'
    })
    logging_steps: int = field(default=1,
                               metadata={
                                   'help': 'steps takes in training to log'
                               })
    logdir: str = field(default='logs',
                        metadata={
                            'help': 'directory to write logs in'
                        })


def setup_model(model_id: str, config=None) -> Tuple[nn.Module, PreTrainedTokenizer]:
    if config is not None:
        model = AutoModelForCausalLM(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def setup():
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def get_date_and_time():
    date_of_run = datetime.now().strftime('%Y-%m-%d-%I:%M:%S_%p')
    print_(f'DATE AND TIME : {date_of_run}')
    return date_of_run


def format_metrics_to_gb(item):
    metric_num = item / (1024 ** 3)
    metric_num = round(metric_num, ndigits=4)
    return metric_num


fp16 = MixedPrecision(
    param_dtype=torch.float16,
    buffer_dtype=torch.float16,
    reduce_dtype=torch.float16
)

bf16 = MixedPrecision(
    param_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16
)

fp32 = MixedPrecision(
    param_dtype=torch.float32,
    buffer_dtype=torch.float32,
    reduce_dtype=torch.float32
)

fp64 = MixedPrecision(
    param_dtype=torch.float64,
    reduce_dtype=torch.float64,
    buffer_dtype=torch.float64
)


class CDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset, num_row):
        self.dataset = dataset
        self.num_row = num_row

    def __len__(self):
        return self.num_row

    def __getitem__(self, item):
        input_ids = torch.tensor(self.dataset[item]['input_ids'])
        attention_mask = torch.tensor(self.dataset[item]['attention_mask'])
        labels = torch.tensor(self.dataset[item]['labels'])
        return input_ids, attention_mask, labels


def sprint(func):
    if os.environ['LOCAL_RANK'] == 0:
        return func
    else:
        return print


@sprint
def print_(*args, **kwargs):
    print(*(f"{arg}" for arg in args), **kwargs)


def train_model(args, model: nn.Module, rank: int, world_size: int, train_loader, optimizer: torch.optim.Optimizer,
                scheduler,
                epoch: int, num_samples: int, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fs_dp_loss = torch.zeros(2).to(local_rank)
    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        in_pbar = tqdm(
            range(num_samples), colour="blue", desc="Training Epoch"
        )
    for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids, attention_mask, labels = make2d(input_ids), make2d(attention_mask), make2d(labels)
        optimizer.zero_grad()

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]

        loss.backward()
        optimizer.step()
        scheduler.step()
        fs_dp_loss[0] += loss.item()
        fs_dp_loss[1] += input_ids.shape[0]
        if rank == 0 and (i + 1) % args.logging_steps == 0:
            in_pbar.update(args.logging_steps)
            in_pbar.set_postfix(loss=loss.item(), learning_rate=scheduler.get_lr())

    dist.all_reduce(fs_dp_loss, op=dist.ReduceOp.SUM)
    loss_train = fs_dp_loss[0] / fs_dp_loss[1]

    if rank == 0:
        in_pbar.close()

        print_(
            f"Train Epoch: \t{epoch}, Loss: \t{loss_train:.4f}"
        )
    return loss_train


def main_process(args: Arguments):
    model, tokenizer = setup_model(args.model_id)

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dataset = load_dataset(args.dataset_name)

    print_(dataset.keys())
    print_("train dataset: ", dataset['train'])
    try:
        print_("validation dataset: ", dataset['validation'])
    except KeyError:
        print_('No validation Data Found')

    def tokenize(x):
        x = tokenizer(x[args.dataset_field], max_length=args.max_sequence_length, truncation=True,
                      return_tensors='pt', padding='max_length', add_special_tokens=False
                      )
        x.input_ids = x.input_ids.view(1, -1)
        x.attention_mask = x.attention_mask.view(1, -1)
        x['labels'] = x.input_ids
        return x

    train_dataset = dataset.map(
        tokenize,
        # batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=32,
        desc=f'Running Tokenize World Size : {world_size}'
    )
    dataset_ = CDataSet(train_dataset['train'], train_dataset['train'].num_rows)
    sampler = DistributedSampler(dataset_, rank=rank, num_replicas=world_size, shuffle=True)

    setup()

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler}
    cuda_kwargs = {'num_workers': world_size,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_,
                                               **train_kwargs)
    # exec(f'from transformers import {args.transformer_cls_to_wrap}')
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # eval(args.transformer_cls_to_wrap)
            args.transformer_cls_to_wrap
        },
    )

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(local_rank)

    bf16_ready = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and dist.is_nccl_available()
            and torch.cuda.nccl.version() >= (2, 10)
    )
    print_(f'BFloat 16 Status {bf16_ready} 😇 ')
    if bf16_ready:
        mp_policy = bf16
    else:
        mp_policy = fp32

    model = FSDP(model,
                 auto_wrap_policy=auto_wrap_policy,
                 mixed_precision=mp_policy, sharding_strategy=sharding_strategy,
                 device_id=torch.cuda.current_device()
                 )
    if rank == 0:
        # print_(f"Model : {model}")
        print_(model)
        print_(f'Model Parameters : {count_model_parameters(model)} Million ')
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_train_steps = math.ceil(train_dataset['train'].num_rows / args.batch_size)
    scheduler = get_scheduler(args.scheduler, optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=num_train_steps * args.epochs)
    file_save_name = "OST-"
    save_dir = Path(args.model_id)
    save_dir.mkdir(exist_ok=True)
    log_dir = Path(f"{save_dir}" + '/' + args.logdir)
    log_dir.mkdir(exist_ok=True)

    if rank == 0:
        time_of_run = get_date_and_time()
        dur = []
        train_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        train_accuracy = train_model(args=args, model=model, rank=rank, train_loader=train_loader, optimizer=optimizer,
                                     scheduler=scheduler,
                                     epoch=epoch, sampler=sampler, world_size=world_size, num_samples=num_train_steps)

        if rank == 0:

            print_(f"[ACTION] >  epoch {epoch} completed...entering save and stats zone")

            train_acc_tracking.append(train_accuracy.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print_(f"completed save and stats zone...")

            # save
            if rank == 0:
                print_(f"[ACTION] >  entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()

            if rank == 0:
                print_(f"[ACTION] >  saving model ...")
                curr_epoch = (
                        "-" + str(epoch) + "-" + str(round(train_accuracy.item(), 4)) + ".pt"
                )
                print_(f"[ACTION] >  attempting to save model prefix {curr_epoch}")
                save_name = file_save_name + time_of_run + "-" + curr_epoch
                print_(f"[ACTION] >  saving as model name {save_name}")

                torch.save(cpu_state, save_name)

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    arguments = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    assert arguments.transformer_cls_to_wrap, 'transformer_cls_to_wrap Can Not be None'
    main_process(arguments)
