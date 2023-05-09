from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer, DataCollatorForLanguageModeling, \
    AutoModelForCausalLM
from datasets import load_dataset
from dataclasses import field, dataclass
import os
import torch
from utils.utils import count_model_parameters
from torch.distributed import is_initialized
import time
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

try:
    import wandb
except ModuleNotFoundError:
    pass
# this will be used for default tokenizer LT and LGeM if you want to train your own model you can skip ahead

LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))

DEFAULT_EOS_TOKEN = '<|endoftext|>'
DEFAULT_BOS_TOKEN = '<|endoftext|>'
DEFAULT_PAD_TOKEN = '<|endoftext|>'
DEFAULT_UNK_TOKEN = '<|endoftext|>'
ADAMW_HF = "adamw_hf"
ADAMW_TORCH = "adamw_torch"
ADAMW_TORCH_FUSED = "adamw_torch_fused"
ADAMW_TORCH_XLA = "adamw_torch_xla"
ADAMW_APEX_FUSED = "adamw_apex_fused"
ADAFACTOR = "adafactor"
ADAMW_BNB = "adamw_bnb_8bit"
ADAMW_ANYPRECISION = "adamw_anyprecision"
SGD = "sgd"
ADAGRAD = "adagrad"

OPTIMIZERS = [
    ADAGRAD,
    ADAMW_HF,
    ADAMW_APEX_FUSED,
    ADAMW_TORCH,
    SGD,
    ADAMW_ANYPRECISION,
    ADAMW_BNB,
    ADAFACTOR,
    ADAMW_TORCH_XLA,
    ADAMW_TORCH_FUSED
]


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


@dataclass
class Arguments:
    # Your Model id Here
    cls_to_wrap: str = field(default='LtBlock', metadata={
        'help': 'transformer layer class to warp for fully sharded data parallel'
    })
    model_id: str = field(default='erfanzar/LT-1B', metadata={
        'help': 'model id to save output and push to hub'
    })
    tokenizer_id: str = field(default='erfanzar/LGeM-7B', metadata={
        'help': 'tokenizer repo id to read from the specific tokenizer that you want to use on model'
    })
    dataset: str = field(default='erfanzar/Base-Data', metadata={
        'help': 'hugging face dataset to download or path to data.json file'
    })
    dataset_field: str = field(default='prompt', metadata={
        'help': 'the specific field ub dataset to look for and run tokenizer on that'
    })
    max_length: int = field(default=1536, metadata={
        'help': 'train max sequence length'
    })
    num_train_epochs: int = field(default=3, metadata={
        'help': 'num train epochs'
    })
    from_safetensors: bool = field(default=True, metadata={
        'help': 'load model from safetensors checkpoints instance of .bin .pt or .pth'
    })
    use_deepspeed: bool = field(default=False, metadata={
        'help': 'use deepspeed for training'
    })
    use_fsdp: bool = field(default=True, metadata={
        'help': 'use fully sharded data parallel for training'
    })
    per_device_batch_size: int = field(default=8, metadata={
        'help': 'pre device batch size'
    })
    auto_batch: bool = field(default=False, metadata={
        'help': 'find batch size automatic'
    })
    learning_rate: float = field(default=1e-4, metadata={
        'help': 'learning rate for the optimzer'
    })
    lr_scheduler_type: str = field(default='cosine', metadata={
        'help': 'learning rate scheduler type type for optimizer'
    })
    do_train: bool = field(default=True, metadata={
        'help': 'do the training or not'
    })
    do_eval: bool = field(default=False, metadata={
        'help': 'do the evaluation or not'
    })
    do_predict: bool = field(default=False, metadata={
        'help': 'do the prediction or not'
    })
    save_safetensors: bool = field(default=True, metadata={
        'help': 'save model in safetensors after training'
    })
    logging_step: int = field(default=15, metadata={
        'help': 'logging steps default to 15'
    })
    report_to: list[str] = field(default='none', metadata={
        'help': 'report training metrics to '
    })
    save_steps: int = field(default=1500, metadata={
        'help': 'steps to save model in training'
    })
    save_strategy: str = field(default='epoch', metadata={
        'help': 'save strategy [epoch or steps]'
    })
    save_total_limit: int = field(default=1, metadata={
        'help': 'total limit of saving model'
    })
    resume_from_checkpoint: bool = field(default=False, metadata={
        'help': 'resume from the last trained checkpoint'
    })
    optimizer: str = field(default='adamw_torch', metadata={
        'help': f'Optimizer to train model available optimizers are {OPTIMIZERS}'
    })
    gradient_accumulation_steps: int = field(default=1, metadata={
        'help': 'gradient accumulation steps'
    })
    do_compile: bool = field(default=False, metadata={
        'help': 'compile the model'
    })
    gradient_checkpointing: bool = field(default=False, metadata={
        'help': 'use gradient checkpointing or not its better to use cause make training better and lighter'
    })


class Timer:

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        started_ = self.started_
        if self.started_:
            self.stop()
        elapsed_ = self.elapsed_
        if reset:
            self.reset()
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):

        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        assert normalizer > 0.0
        string = "time (ms)"
        if isinstance(names, str):
            names = [names]
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}  [RANK : {} / {}]:  {:.2f}".format(name, LOCAL_RANK + 1, WORLD_SIZE, elapsed_time)
        if is_initialized():
            if LOCAL_RANK == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def check_tokenizer(tokenizer):
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


def main(args: Arguments):
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
    log_t_path = Path('out\\performance_metrics')
    log_t_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=f'{log_t_path}')
    timers = Timers(
        use_wandb=False,
        tensorboard_writer=writer
    )
    timers('getting tokenizer').start()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer = check_tokenizer(tokenizer)
    timers('getting tokenizer').stop()
    timers.log('getting tokenizer')
    timers('building model ...').start()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True,
                                                 from_safetensors=args.from_safetensors)

    timers('building model ...').stop()

    timers.log('building model ...')

    timers('getting data').start()

    dataset = load_dataset(args.dataset)

    timers('getting data').stop()
    timers.log('getting data')

    timers('mapping data').start()
    dataset = dataset.map(
        lambda data_point: tokenizer(data_point[args.dataset_field], max_length=args.max_length, padding='max_length',
                                     truncation=True,
                                     add_special_tokens=False))
    timers('mapping data').stop()
    timers.log('mapping data')
    timers('creat or eval training arguments').start()

    assert args.optimizer in OPTIMIZERS, f'invalid optimizer {args.optimizer}'
    if args.use_deepspeed:
        extra_kwargs = {
            'deepspeed': DEEPSPEED_CONFIG
        }
    elif args.use_fsdp:
        extra_kwargs = (
            dict(
                fsdp='auto_wrap full_shard',
                fsdp_config={
                    'fsdp_transformer_layer_cls_to_wrap': {args.cls_to_wrap}
                }
            )
        )

    else:
        print_rank_0('ENGINE DIDNT GET ANY FSDP CONFIG OR DEEPSPEED - SKIP')
        extra_kwargs = {}

    if args.do_eval:
        try:
            eval_dataset = dataset['valid']
        except KeyError:
            print_rank_0('NO EVALUATION DATA FOUND SETTING DO EVAL TO FALSE INCASE IGNORE ANY BUG DURING TRAINING')
            args.do_eval = False
            eval_dataset = None
    else:
        eval_dataset = None
    try:
        model.transformer.gradient_checkpointing = True
    except:
        model.model.gradient_checkpointing = True
    training_args = TrainingArguments(
        output_dir=args.model_id,
        hub_model_id=args.model_id,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        auto_find_batch_size=args.auto_batch,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_eval,
        per_device_train_batch_size=args.per_device_batch_size,
        logging_steps=args.logging_step,
        logging_dir=f'{args.model_id}/logs',
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        seed=42,
        # fp16=True,
        optim=args.optimizer,
        weight_decay=1e-2,
        report_to=args.report_to if args.report_to is not None else 'none',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_safetensors=args.save_safetensors,
        num_train_epochs=args.num_train_epochs,
        torch_compile=args.do_compile,
        gradient_checkpointing=False,
        **extra_kwargs,

    )
    print_rank_0('MODEL CONTAIN ', count_model_parameters(model, 1e9), ' BILLION PARAMETERS ')
    timers('creat or eval training arguments').stop()
    timers.log('creat or eval training arguments')
    model.model.gradient_checkpointing = args.gradient_checkpointing

    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors='pt'
        )
    )
    print_rank_0(f'MODEL CONTAIN {count_model_parameters(model, div=1e9)} BILLION PARAMETERS')
    timers('training model').start()
    timers.write(
        ['training model', 'creat or eval training arguments', 'mapping data', 'getting data', 'building model ...',
         'getting tokenizer'], 0)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    timers('training model').stop()
    timers.log('training model')
    trainer.push_to_hub(f'Done Training {args.model_id} for {args.num_train_epochs} ')


if __name__ == "__main__":
    args_: Arguments = HfArgumentParser((Arguments,)).parse_args_into_dataclasses()[0]
    # print_rank_0(args_)
    main(args_)
