# Note
# the GPT-j implemented by Huggingface is not supporting Partition spec s and using
# fully with pjit and required creating
# parameters even if you want to load already trained model so this one is the same
# but include those bugs / non-features
# fixed

from .modelling_gpt_j import FlaxGPTJModel, FlaxGPTJForCausalLM, GPTJConfig
