import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, logging, \
    pipeline

## THIS FILE IS TODO

import textwrap
import os
import datetime
from dataclasses import field, dataclass
from transformers import HfArgumentParser
import gradio as gr
import whisper
from fjutils.utils import count_num_params
from fjutils.easylm import with_sharding_constraint, make_shard_and_gather_fns, match_partition_rules
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.serialization import from_bytes, to_state_dict, from_state_dict

logger = logging.get_logger(__name__)


@dataclass
class LoadConfig:
    ckpt_path: str = field()
    mode: str = field(default='gui-chat', metadata={'help': 'mode to use ai in '})
    model_id: str = field(default='erfanzar/LGeM-7B-C', metadata={'help': 'model to load'})
    load_from_hf: bool = field(default=False)
    whisper_model: str = field(default='base', metadata={'help': 'model to load for whisper '})
    use_lgem_stoper: bool = field(default=False)


def load_model(config_: LoadConfig):
    tokenizer = AutoTokenizer.from_pretrained(config_.model_id)
    # with open(config_.ckpt_path, 'rb') as stream:
    ...


if __name__ == "__main__":
    config = HfArgumentParser(LoadConfig).parse_args_into_dataclasses()[0]
    model_, params, tokenizer_ = load_model(config)
