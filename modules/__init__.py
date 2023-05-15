import os

try:
    _ = os.environ['USE_JIT']
except KeyError:
    os.environ['USE_JIT'] = '0'

from .datasets import Tokens
# from .jax_modeling_PPaLM import PPaLM, PPaLMConfig
# PPaLM is an edited version of PaLM from Google but edited at some points
# try:
# from .jax_modelling_flax_LGeM import LGemModel, LGemModelForCasualLM, LGemConfig
from modules.modelling_lgem.modeling_LGeM import LGeMModel, LGeMConfig, LGeMForCausalLM
from modules.modelling_llmou.modeling_LLMoU import LLMoUModel, LLMoUConfig
from modules.modelling_llama.modeling_LLaMA import LLamaModel, LLamaConfig
from modules.modelling_llmpu.modeling_LLmPU import LLmPUForConditionalGeneration, LLmPUModel, LLmPUConfig
from modules.modelling_rwkv.modelling_RWKV import RWKVConfig, RWKV_GPT_CasualLM, RWKVConfigTrain
from modules.modelling_pgt import PGT, PGTConfig, PGTForCausalLM
from modules.modelling_llmp import LLmP, LLmPConfig
from modules.modelling_lt import LtModel, LtModelForCausalLM, LtConfig
from modules.triton import FlashAttnKVPackedFunc, flash_attn_kvpacked_func, flash_attn_qkvpacked_func, \
    FlashAttnQKVPackedFunc, FlashAttnFunc, _flash_attn_forward
from modules.modelling_palm import PalmConfig, PalmModel, PalmForCausalLM
# except ImportError:
#     pass

from modules.jax_modules import FlaxLGeMModule, FlaxLGeMConfig, FlaxLGeMModel, FlaxLGeMForCausalLM
