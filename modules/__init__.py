import os

try:
    _ = os.environ['USE_JIT']
except KeyError:
    os.environ['USE_JIT'] = '0'

from .datasets import Tokens

from modules.triton import FlashAttnKVPackedFunc, flash_attn_kvpacked_func, flash_attn_qkvpacked_func, \
    FlashAttnQKVPackedFunc, FlashAttnFunc, _flash_attn_forward
from modules.pytorch_modules import PalmConfig, PalmModel, PalmForCausalLM, LtModel, LtModelForCausalLM, LtConfig, \
    LLmPUForConditionalGeneration, LLmPUModel, \
    LLmPUConfig, LLMoUModel, LLMoUConfig, LGeMModel, LGeMConfig, LGeMForCausalLM

from modules.jax_modules import FlaxAGeMModule, FlaxAGeMConfig, FlaxAGeMModel, FlaxAGeMForCausalLM, \
    FlaxLTForCausalLM, FlaxLTPretrainedModel, FlaxLTModel, FlaxLTConfig, FlaxGPTJForCausalLM, \
    FlaxGPTJModel, GPTJConfig, FlaxLGeMConfig, FlaxLGeMModel, FlaxLGeMForCausalLM
