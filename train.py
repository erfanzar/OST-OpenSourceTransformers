from modules import LtConfig, LtModelForCausalLM
from transformers import Trainer, TrainingArguments, HfArgumentParser, LlamaTokenizer
from transformers.training_args import OptimizerNames
from datasets import load_dataset
from dataclasses import field, dataclass
import os
import torch
from utils.utils import count_model_parameters

# this will be used for default tokenizer LT and LGeM if you want to train your own model you can skip ahead

LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))

DEFAULT_EOS_TOKEN = '<|endoftext|>'
DEFAULT_BOS_TOKEN = '<|endoftext|>'
DEFAULT_PAD_TOKEN = '<|endoftext|>'
DEFAULT_UNK_TOKEN = '<|endoftext|>'


def print_rank_0(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)


EXTRA_TOKENS = [
    '<|system|>',
    '<|assistant|>',
    '<|prompter|>'
]

DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "offload_optimizer": {
            "device": "cpu"
        },
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 0,
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
    "offload_optimizer": {
        "device": "cpu"
    },
    "offload_param": {
        "device": "cpu",
        "nvme_path": "/local_nvme",
        "pin_memory": False,
        "buffer_count": 5,
        "buffer_size": 1e8,
        "max_in_cpu": 1e9
    }
}


def check_tokenizer(tokenizer: LlamaTokenizer):
    print_rank_0('CHANGING TOKENIZER OPTIONS')
    tokenizer.pad_token = DEFAULT_PAD_TOKEN
    tokenizer.bos_token = DEFAULT_BOS_TOKEN
    tokenizer.eos_token = DEFAULT_EOS_TOKEN
    tokenizer.unk_token = DEFAULT_UNK_TOKEN

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.unk_token_id = tokenizer.eos_token_id

    tokenizer.add_tokens(EXTRA_TOKENS)

    print_rank_0('DONE - CHANGING TOKENIZER OPTIONS')
    return tokenizer


@dataclass
class Arguments:
    # Your Model id Here
    cls_to_wrap: str = field()
    model_id: str = field(default='erfanzar/LT-1B')
    tokenizer_id: str = field(default='erfanzar/LGeM-7B')
    dataset: str = field(default='erfanzar/Base-Data')
    dataset_field: str = field(default='prompt')
    max_length: int = field(default=1536)
    from_safetensors: bool = field(default=True)
    lr_sc: str = field(default='cosine')
    use_deepspeed: bool = field(default=False)
    use_fsdp: bool = field(default=True)
    per_device_batch_size: int = field(default=8)
    auto_batch: bool = field(default=False)
    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(default='cosine')
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    logging_step: int = field(default=15)
    report_to: list[str] = field(default='tensorboard')
    save_steps: int = field(default=1500)
    save_strategy: str = field(default='epoch')
    save_total_limit: int = field(default=1)
    resume_from_checkpoint: bool = field(default=False)


def main():
    args: Arguments = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    if args.use_deepspeed:
        tip = f'deepspeed --no_python --master_addr=4008 --num_gpus={torch.cuda.device_count()} train.py *your-args'
    elif args.use_fsdp:
        tip = f'torchrun --nproc-per-node={torch.cuda.device_count()} --master-port=4008 --standalone' \
              f' train.py *your-args'
    else:
        tip = 'NO FSPD OR DEEPSPEED SELECTED'
    print_rank_0(
        f'TIP : You are using '
        f'{"DEEPSPEED TRAGEDY" if args.use_deepspeed else ("Fully Sharded Data Parallel" if args.use_fsdp else "No Option")}'
        f'\nRecommended Run command => \n\t {tip} '
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer = check_tokenizer(tokenizer)
    config = LtConfig(vocab_size=len(tokenizer.get_vocab()), num_attention_heads=8, num_hidden_layers=4,
                      hidden_size=256,
                      intermediate_size=512)
    model = LtModelForCausalLM(config=config)
    dataset = load_dataset(args.dataset)

    dataset = dataset.map(
        lambda data_point: tokenizer(data_point[args.dataset_field], max_length=args.max_length, padding='max_length',
                                     truncation=True,
                                     add_special_tokens=False))

    if args.use_deepspeed:
        extra_kwargs = {
            'deepspeed': DEEPSPEED_CONFIG
        }
    elif args.use_fsdp:
        extra_kwargs = (
            dict(
                fsdp='auto_wrap full_shard',
                fsdp_config={
                    'fsdp_transformer_layer_cls_to_wrap': args.cls_to_wrap
                }
            )
        )

    else:
        print_rank_0('ENGINE DIDNT GET ANY FSDP CONFIG OR DEEPSPEED - SKIP')
        extra_kwargs = {}
    training_args = TrainingArguments(
        output_dir=args.model_id,
        hub_model_id=args.model_id,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        auto_find_batch_size=args.auto_batch,
        do_train=True,
        do_eval=False,
        do_predict=True,
        per_device_train_batch_size=args.per_device_batch_size,
        logging_steps=args.logging_step,
        logging_dir=f'{args.model_id}/logs',
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        seed=42,
        fp16=True,
        optim=OptimizerNames.ADAMW_TORCH,
        weight_decay=1e-2,
        report_to=['tensorboard'],
        save_safetensors=args.save_safetensors,
        **extra_kwargs,

    )

    print_rank_0(model)

    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        args=training_args
    )
    print_rank_0(f'MODEL CONTAIN {count_model_parameters(model, div=1e9)} BILLION PARAMETERS')
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
