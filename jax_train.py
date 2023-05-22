import jax
from modules import FlaxLGeMForCausalLM, FlaxLGeMConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, AutoConfig
from jax import numpy as jnp

LOCAL_MODELS = [
    'PaLM',
    'LGeM'
]


@dataclass
class TrainingArguments:
    output_dir: str = field(
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
    model_name: Optional[str] = field(
        default='LGeM',
        metadata={
            "help":
                "Model Name Or Type"

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
    arguments = HfArgumentParser((TrainingArguments, ModelArguments, DataTrainingArguments))
    train_args, model_args, data_args = arguments.parse_args_into_dataclasses()
    train_args: TrainingArguments = train_args
    model_args: ModelArguments = model_args
    data_args: DataTrainingArguments = data_args
    config = AutoConfig.from_pretrained(model_args.config_name) if model_args.config_name is not None else None

    if model_args.model_name in LOCAL_MODELS:
        if model_args.model_name == 'LGeM':
            if config is None:
                assert model_args.model_name
                model = FlaxLGeMForCausalLM.from_pretrained(
                    model_args.model_name,
                    use_auth_token=model_args.use_auth_token,
                    dtype=jnp.dtype(model_args.dtype),
                    trust_remote_code=model_args.trust_remote_code
                )
            else:
                model = FlaxLGeMForCausalLM(
                    config=config
                )


if __name__ == "__main__":
    main()
