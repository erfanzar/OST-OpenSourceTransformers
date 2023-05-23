import erutils
import flax.core
import jax
from modules import FlaxLGeMForCausalLM, FlaxLGeMConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, AutoConfig, FlaxAutoModelForCausalLM, AutoTokenizer, FlaxPreTrainedModel, \
    logging

from jax import numpy as jnp
from utils.timer import Timers

logging.set_verbosity_error()
# jax.config.update()
jax.default_device = jax.devices('cpu')[0]


def prefix_printer(prefix, value):
    print(f' \033[1;31m{prefix}\033[1;0m : {value}')


def get_model_devices(params) -> [str]:
    devi = {}
    for pd in jax.tree_util.tree_flatten(flax.core.unfreeze(params))[0]:
        if isinstance(pd, jnp.DeviceArray):
            devi[pd.device()] = 'Using'
    return list(devi.keys())


def check_device():
    print('Device Checking in order ...')
    dev = {
        'CPU': jax.devices('cpu'),
        'GPU': jax.devices('gpu')
    }
    none_val_tpu = 'NOT found'
    try:
        dev += dict(TPU=jax.devices('tpu'))
    except RuntimeError:

        dev['TPU'] = none_val_tpu
    print(f'founded Accelerators on this device are ')

    for k, v in dev.items():
        prefix_printer(k, v)
    gpu = jax.devices('gpu')
    tpu = dev['TPU']
    prefix_printer('Device Report', f'This machine Contain {len(dev["CPU"])}'
                                    f' CPUS , {f"have {len(gpu)} GPUS " if len(gpu) != 0 else "have No GPU"} and '
                                    f'{f"have {len(tpu)} TPU Core" if tpu != none_val_tpu else "have No TPU Core"}')
    return dev


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='OST/JaxModel',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    per_device_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})

    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})

    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})

    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )

    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )

    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."

        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path "}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Trust remote code or Not ? "
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_field: Optional[str] = field(
        default=None, metadata={"help": "The field in data set to run tokenizer on that field"}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


def main():
    check_device()
    arguments = HfArgumentParser((TrainingArguments, ModelArguments, DataTrainingArguments))
    train_args, model_args, data_args = arguments.parse_args_into_dataclasses()
    train_args: TrainingArguments = train_args
    model_args: ModelArguments = model_args
    data_args: DataTrainingArguments = data_args

    memory_flag = {'tokenizer_name': model_args.tokenizer_name, 'model_name': model_args.model_name_or_path,
                   'config_from': model_args.config_name, 'dataset_name': data_args.dataset_name,
                   'dataset_field': data_args.dataset_field, 'block_size|max_sequence_length': data_args.block_size,
                   'num_train_epochs': train_args.num_train_epochs, 'seed': train_args.seed,
                   'hub_model_id': train_args.hub_model_id, 'save_steps': train_args.save_steps,
                   'logging_steps': train_args.logging_steps, 'learning_rate': train_args.learning_rate}

    assert model_args.tokenizer_name, 'tokenizer_name is a required field for trainer ' \
                                      'please pass a tokenizer path or repo id'
    prefix_printer('Default Device Jax', jax.default_device)
    config = AutoConfig.from_pretrained(model_args.config_name,
                                        trust_remote_code=True,

                                        ) if model_args.config_name is not None else None
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True)
    assert len(tokenizer.get_vocab()) == config.vocab_size

    if config is None:
        assert model_args.model_name_or_path
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_auth_token=model_args.use_auth_token,
            dtype=jnp.dtype(model_args.dtype),
            trust_remote_code=model_args.trust_remote_code,
            # revision='main'
        )
    else:
        model: flax.linen.Module | FlaxPreTrainedModel = FlaxAutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            # revision='main'
        )

    _i = jax.tree_util.tree_flatten(flax.core.unfreeze(model.params))[0]
    prefix_printer('Model Contain ', f'{sum(i.size for i in _i) / 1e6} Million Parameters')
    prefix_printer('Model Devices Are', f'{get_model_devices(model.params)}')


if __name__ == "__main__":
    main()
