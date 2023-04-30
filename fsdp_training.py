from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType
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
from typing import Type
import torch.distributed as dist
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, logging

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

model_id_ = 'bigscience/bigscience-small-testing'
scheduler_name = 'cosine'
gigabyte = 1024 ** 3


def setup_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer


def setup():
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def get_date_and_time():
    date_of_run = datetime.now().strftime('%Y-%m-%d-%I:%M:%S_%p')
    logger.info(f'DATE AND TIME : {date_of_run}')
    return date_of_run


def format_metrics_to_gb(item):
    metric_num = item / gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def train(args, model: torch.nn.Module, rank: int, world_size: int, train_loader, optimizer: torch.optim.Optimizer,
          epochs: int, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fs_dp_loss = torch.zeros(2).to(local_rank)
    if sampler:
        sampler.set_epoch(epochs)
    if rank == 0:
        in_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fs_dp_loss[0] += loss.item()
        fs_dp_loss[1] += len(batch)
        if rank == 0:
            in_pbar.update(1)

    dist.all_reduce(fs_dp_loss, op=dist.ReduceOp.SUM)
    loss_train = fs_dp_loss[0] / fs_dp_loss[1]

    if rank == 0:
        in_pbar.close()
        print(
            f"Train Epoch: \t{epochs}, Loss: \t{loss_train:.4f}"
        )
    return loss_train


def fsdp_main(args):
    model, tokenizer = setup_model("t5-base")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dataset = load_dataset()
    print(dataset.keys())
    print("Size of train dataset: ", dataset['train'].shape)
    print("Size of Validation dataset: ", dataset['validation'].shape)
    # TODO
    train_dataset = dataset.map()

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)

    setup()

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP  # for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)

    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    bf16_ready = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None  # defaults to fp32

    model = FSDP(model,
                 auto_wrap_policy=auto_wrap_policy,
                 mixed_precision=mp_policy,
                 # sharding_strategy=sharding_strategy,
                 device_id=torch.cuda.current_device())

    optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = get_scheduler(args.scheduler_mode, optimizer=optimizer, num_warmup_steps=0)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "FSDP_Trained_"

    if rank == 0:
        time_of_run = get_date_and_time()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)

        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")

        # init_end_event.record()

        # if rank == 0:
        # print(f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # print(f"{model}")

        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            # print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                        "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()
