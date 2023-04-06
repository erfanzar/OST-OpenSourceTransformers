import logging
import os

import accelerate
import torch
from torch import nn
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

os.environ['USE_JIT'] = "0"

from modules import LGeMConfig, LGeMForCausalLM

torch.set_default_dtype(torch.float16)
logger.info('Start Loading CKPT')
data = torch.load('E:/Checkpoints/LGeM/LGeM.pt')
logger.info('End Loading CKPT')
config = LGeMConfig(hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32,
                    vocab_size=32000, max_sentence_length=64)
tokenizer = PreTrainedTokenizer.from_pretrained('tokenizer_model/BASE')

with accelerate.init_empty_weights():
    logger.info('Creating Model Started')
    model = LGeMForCausalLM(config)
    logger.info('Created')


def load_lgm_weights(model, weights, requires_grad=True):
    model_embed_tokens_weight = weights['model.embed_tokens.weight']
    model.model.embed_tokens.weight = torch.nn.Parameter(model_embed_tokens_weight)

    for i, layer in enumerate(model.model.layers):

        layer.self_attn.q_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.self_attn.q_proj.weight'],
                                                           requires_grad=requires_grad)
        layer.self_attn.k_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.self_attn.k_proj.weight'],
                                                           requires_grad=requires_grad)
        layer.self_attn.v_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.self_attn.v_proj.weight'],
                                                           requires_grad=requires_grad)
        layer.self_attn.o_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.self_attn.o_proj.weight'],
                                                           requires_grad=requires_grad)
        if hasattr(layer.self_attn, 'rotary_emb'):
            layer.self_attn.rotary_emb.inv_freq = torch.nn.Parameter(
                weights[f'model.layers.{i}.self_attn.rotary_emb.inv_freq'], requires_grad=requires_grad)
        layer.mlp.down_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.mlp.down_proj.weight'],
                                                        requires_grad=requires_grad)
        layer.mlp.gate_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.mlp.gate_proj.weight'],
                                                        requires_grad=requires_grad)
        layer.mlp.up_proj.weight = torch.nn.Parameter(weights[f'model.layers.{i}.mlp.up_proj.weight'],
                                                      requires_grad=requires_grad)
        layer.input_layernorm.weight = torch.nn.Parameter(weights[f'model.layers.{i}.input_layernorm.weight'],
                                                          requires_grad=requires_grad)
        layer.post_attention_layernorm.weight = torch.nn.Parameter(
            weights[f'model.layers.{i}.post_attention_layernorm.weight'], requires_grad=requires_grad)
    model.model.norm.weight = torch.nn.Parameter(weights[f'model.norm.weight'])
    model.lm_head.weight = torch.nn.Parameter(weights[f'lm_head.weight'])
    return model


# Generated Code
STATIC = False
if STATIC:
    model.model.embed_tokens.weight = nn.Parameter(data['model.embed_tokens.weight'])
    model.model.layers[0].self_attn.q_proj.weight = nn.Parameter(data['model.layers.0.self_attn.q_proj.weight'])
    model.model.layers[0].self_attn.k_proj.weight = nn.Parameter(data['model.layers.0.self_attn.k_proj.weight'])
    model.model.layers[0].self_attn.v_proj.weight = nn.Parameter(data['model.layers.0.self_attn.v_proj.weight'])
    model.model.layers[0].self_attn.o_proj.weight = nn.Parameter(data['model.layers.0.self_attn.o_proj.weight'])
    model.model.layers[0].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.0.self_attn.rotary_emb.inv_freq'])
    model.model.layers[0].mlp.gate_proj.weight = nn.Parameter(data['model.layers.0.mlp.gate_proj.weight'])
    model.model.layers[0].mlp.down_proj.weight = nn.Parameter(data['model.layers.0.mlp.down_proj.weight'])
    model.model.layers[0].mlp.up_proj.weight = nn.Parameter(data['model.layers.0.mlp.up_proj.weight'])
    model.model.layers[0].input_layernorm.weight = nn.Parameter(data['model.layers.0.input_layernorm.weight'])
    model.model.layers[0].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.0.post_attention_layernorm.weight'])
    model.model.layers[1].self_attn.q_proj.weight = nn.Parameter(data['model.layers.1.self_attn.q_proj.weight'])
    model.model.layers[1].self_attn.k_proj.weight = nn.Parameter(data['model.layers.1.self_attn.k_proj.weight'])
    model.model.layers[1].self_attn.v_proj.weight = nn.Parameter(data['model.layers.1.self_attn.v_proj.weight'])
    model.model.layers[1].self_attn.o_proj.weight = nn.Parameter(data['model.layers.1.self_attn.o_proj.weight'])
    model.model.layers[1].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.1.self_attn.rotary_emb.inv_freq'])
    model.model.layers[1].mlp.gate_proj.weight = nn.Parameter(data['model.layers.1.mlp.gate_proj.weight'])
    model.model.layers[1].mlp.down_proj.weight = nn.Parameter(data['model.layers.1.mlp.down_proj.weight'])
    model.model.layers[1].mlp.up_proj.weight = nn.Parameter(data['model.layers.1.mlp.up_proj.weight'])
    model.model.layers[1].input_layernorm.weight = nn.Parameter(data['model.layers.1.input_layernorm.weight'])
    model.model.layers[1].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.1.post_attention_layernorm.weight'])
    model.model.layers[2].self_attn.q_proj.weight = nn.Parameter(data['model.layers.2.self_attn.q_proj.weight'])
    model.model.layers[2].self_attn.k_proj.weight = nn.Parameter(data['model.layers.2.self_attn.k_proj.weight'])
    model.model.layers[2].self_attn.v_proj.weight = nn.Parameter(data['model.layers.2.self_attn.v_proj.weight'])
    model.model.layers[2].self_attn.o_proj.weight = nn.Parameter(data['model.layers.2.self_attn.o_proj.weight'])
    model.model.layers[2].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.2.self_attn.rotary_emb.inv_freq'])
    model.model.layers[2].mlp.gate_proj.weight = nn.Parameter(data['model.layers.2.mlp.gate_proj.weight'])
    model.model.layers[2].mlp.down_proj.weight = nn.Parameter(data['model.layers.2.mlp.down_proj.weight'])
    model.model.layers[2].mlp.up_proj.weight = nn.Parameter(data['model.layers.2.mlp.up_proj.weight'])
    model.model.layers[2].input_layernorm.weight = nn.Parameter(data['model.layers.2.input_layernorm.weight'])
    model.model.layers[2].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.2.post_attention_layernorm.weight'])
    model.model.layers[3].self_attn.q_proj.weight = nn.Parameter(data['model.layers.3.self_attn.q_proj.weight'])
    model.model.layers[3].self_attn.k_proj.weight = nn.Parameter(data['model.layers.3.self_attn.k_proj.weight'])
    model.model.layers[3].self_attn.v_proj.weight = nn.Parameter(data['model.layers.3.self_attn.v_proj.weight'])
    model.model.layers[3].self_attn.o_proj.weight = nn.Parameter(data['model.layers.3.self_attn.o_proj.weight'])
    model.model.layers[3].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.3.self_attn.rotary_emb.inv_freq'])
    model.model.layers[3].mlp.gate_proj.weight = nn.Parameter(data['model.layers.3.mlp.gate_proj.weight'])
    model.model.layers[3].mlp.down_proj.weight = nn.Parameter(data['model.layers.3.mlp.down_proj.weight'])
    model.model.layers[3].mlp.up_proj.weight = nn.Parameter(data['model.layers.3.mlp.up_proj.weight'])
    model.model.layers[3].input_layernorm.weight = nn.Parameter(data['model.layers.3.input_layernorm.weight'])
    model.model.layers[3].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.3.post_attention_layernorm.weight'])
    model.model.layers[4].self_attn.q_proj.weight = nn.Parameter(data['model.layers.4.self_attn.q_proj.weight'])
    model.model.layers[4].self_attn.k_proj.weight = nn.Parameter(data['model.layers.4.self_attn.k_proj.weight'])
    model.model.layers[4].self_attn.v_proj.weight = nn.Parameter(data['model.layers.4.self_attn.v_proj.weight'])
    model.model.layers[4].self_attn.o_proj.weight = nn.Parameter(data['model.layers.4.self_attn.o_proj.weight'])
    model.model.layers[4].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.4.self_attn.rotary_emb.inv_freq'])
    model.model.layers[4].mlp.gate_proj.weight = nn.Parameter(data['model.layers.4.mlp.gate_proj.weight'])
    model.model.layers[4].mlp.down_proj.weight = nn.Parameter(data['model.layers.4.mlp.down_proj.weight'])
    model.model.layers[4].mlp.up_proj.weight = nn.Parameter(data['model.layers.4.mlp.up_proj.weight'])
    model.model.layers[4].input_layernorm.weight = nn.Parameter(data['model.layers.4.input_layernorm.weight'])
    model.model.layers[4].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.4.post_attention_layernorm.weight'])
    model.model.layers[5].self_attn.q_proj.weight = nn.Parameter(data['model.layers.5.self_attn.q_proj.weight'])
    model.model.layers[5].self_attn.k_proj.weight = nn.Parameter(data['model.layers.5.self_attn.k_proj.weight'])
    model.model.layers[5].self_attn.v_proj.weight = nn.Parameter(data['model.layers.5.self_attn.v_proj.weight'])
    model.model.layers[5].self_attn.o_proj.weight = nn.Parameter(data['model.layers.5.self_attn.o_proj.weight'])
    model.model.layers[5].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.5.self_attn.rotary_emb.inv_freq'])
    model.model.layers[5].mlp.gate_proj.weight = nn.Parameter(data['model.layers.5.mlp.gate_proj.weight'])
    model.model.layers[5].mlp.down_proj.weight = nn.Parameter(data['model.layers.5.mlp.down_proj.weight'])
    model.model.layers[5].mlp.up_proj.weight = nn.Parameter(data['model.layers.5.mlp.up_proj.weight'])
    model.model.layers[5].input_layernorm.weight = nn.Parameter(data['model.layers.5.input_layernorm.weight'])
    model.model.layers[5].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.5.post_attention_layernorm.weight'])
    model.model.layers[6].self_attn.q_proj.weight = nn.Parameter(data['model.layers.6.self_attn.q_proj.weight'])
    model.model.layers[6].self_attn.k_proj.weight = nn.Parameter(data['model.layers.6.self_attn.k_proj.weight'])
    model.model.layers[6].self_attn.v_proj.weight = nn.Parameter(data['model.layers.6.self_attn.v_proj.weight'])
    model.model.layers[6].self_attn.o_proj.weight = nn.Parameter(data['model.layers.6.self_attn.o_proj.weight'])
    model.model.layers[6].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.6.self_attn.rotary_emb.inv_freq'])
    model.model.layers[6].mlp.gate_proj.weight = nn.Parameter(data['model.layers.6.mlp.gate_proj.weight'])
    model.model.layers[6].mlp.down_proj.weight = nn.Parameter(data['model.layers.6.mlp.down_proj.weight'])
    model.model.layers[6].mlp.up_proj.weight = nn.Parameter(data['model.layers.6.mlp.up_proj.weight'])
    model.model.layers[6].input_layernorm.weight = nn.Parameter(data['model.layers.6.input_layernorm.weight'])
    model.model.layers[6].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.6.post_attention_layernorm.weight'])
    model.model.layers[7].self_attn.q_proj.weight = nn.Parameter(data['model.layers.7.self_attn.q_proj.weight'])
    model.model.layers[7].self_attn.k_proj.weight = nn.Parameter(data['model.layers.7.self_attn.k_proj.weight'])
    model.model.layers[7].self_attn.v_proj.weight = nn.Parameter(data['model.layers.7.self_attn.v_proj.weight'])
    model.model.layers[7].self_attn.o_proj.weight = nn.Parameter(data['model.layers.7.self_attn.o_proj.weight'])
    model.model.layers[7].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.7.self_attn.rotary_emb.inv_freq'])
    model.model.layers[7].mlp.gate_proj.weight = nn.Parameter(data['model.layers.7.mlp.gate_proj.weight'])
    model.model.layers[7].mlp.down_proj.weight = nn.Parameter(data['model.layers.7.mlp.down_proj.weight'])
    model.model.layers[7].mlp.up_proj.weight = nn.Parameter(data['model.layers.7.mlp.up_proj.weight'])
    model.model.layers[7].input_layernorm.weight = nn.Parameter(data['model.layers.7.input_layernorm.weight'])
    model.model.layers[7].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.7.post_attention_layernorm.weight'])
    model.model.layers[8].self_attn.q_proj.weight = nn.Parameter(data['model.layers.8.self_attn.q_proj.weight'])
    model.model.layers[8].self_attn.k_proj.weight = nn.Parameter(data['model.layers.8.self_attn.k_proj.weight'])
    model.model.layers[8].self_attn.v_proj.weight = nn.Parameter(data['model.layers.8.self_attn.v_proj.weight'])
    model.model.layers[8].self_attn.o_proj.weight = nn.Parameter(data['model.layers.8.self_attn.o_proj.weight'])
    model.model.layers[8].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.8.self_attn.rotary_emb.inv_freq'])
    model.model.layers[8].mlp.gate_proj.weight = nn.Parameter(data['model.layers.8.mlp.gate_proj.weight'])
    model.model.layers[8].mlp.down_proj.weight = nn.Parameter(data['model.layers.8.mlp.down_proj.weight'])
    model.model.layers[8].mlp.up_proj.weight = nn.Parameter(data['model.layers.8.mlp.up_proj.weight'])
    model.model.layers[8].input_layernorm.weight = nn.Parameter(data['model.layers.8.input_layernorm.weight'])
    model.model.layers[8].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.8.post_attention_layernorm.weight'])
    model.model.layers[9].self_attn.q_proj.weight = nn.Parameter(data['model.layers.9.self_attn.q_proj.weight'])
    model.model.layers[9].self_attn.k_proj.weight = nn.Parameter(data['model.layers.9.self_attn.k_proj.weight'])
    model.model.layers[9].self_attn.v_proj.weight = nn.Parameter(data['model.layers.9.self_attn.v_proj.weight'])
    model.model.layers[9].self_attn.o_proj.weight = nn.Parameter(data['model.layers.9.self_attn.o_proj.weight'])
    model.model.layers[9].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.9.self_attn.rotary_emb.inv_freq'])
    model.model.layers[9].mlp.gate_proj.weight = nn.Parameter(data['model.layers.9.mlp.gate_proj.weight'])
    model.model.layers[9].mlp.down_proj.weight = nn.Parameter(data['model.layers.9.mlp.down_proj.weight'])
    model.model.layers[9].mlp.up_proj.weight = nn.Parameter(data['model.layers.9.mlp.up_proj.weight'])
    model.model.layers[9].input_layernorm.weight = nn.Parameter(data['model.layers.9.input_layernorm.weight'])
    model.model.layers[9].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.9.post_attention_layernorm.weight'])
    model.model.layers[10].self_attn.q_proj.weight = nn.Parameter(data['model.layers.10.self_attn.q_proj.weight'])
    model.model.layers[10].self_attn.k_proj.weight = nn.Parameter(data['model.layers.10.self_attn.k_proj.weight'])
    model.model.layers[10].self_attn.v_proj.weight = nn.Parameter(data['model.layers.10.self_attn.v_proj.weight'])
    model.model.layers[10].self_attn.o_proj.weight = nn.Parameter(data['model.layers.10.self_attn.o_proj.weight'])
    model.model.layers[10].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.10.self_attn.rotary_emb.inv_freq'])
    model.model.layers[10].mlp.gate_proj.weight = nn.Parameter(data['model.layers.10.mlp.gate_proj.weight'])
    model.model.layers[10].mlp.down_proj.weight = nn.Parameter(data['model.layers.10.mlp.down_proj.weight'])
    model.model.layers[10].mlp.up_proj.weight = nn.Parameter(data['model.layers.10.mlp.up_proj.weight'])
    model.model.layers[10].input_layernorm.weight = nn.Parameter(data['model.layers.10.input_layernorm.weight'])
    model.model.layers[10].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.10.post_attention_layernorm.weight'])
    model.model.layers[11].self_attn.q_proj.weight = nn.Parameter(data['model.layers.11.self_attn.q_proj.weight'])
    model.model.layers[11].self_attn.k_proj.weight = nn.Parameter(data['model.layers.11.self_attn.k_proj.weight'])
    model.model.layers[11].self_attn.v_proj.weight = nn.Parameter(data['model.layers.11.self_attn.v_proj.weight'])
    model.model.layers[11].self_attn.o_proj.weight = nn.Parameter(data['model.layers.11.self_attn.o_proj.weight'])
    model.model.layers[11].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.11.self_attn.rotary_emb.inv_freq'])
    model.model.layers[11].mlp.gate_proj.weight = nn.Parameter(data['model.layers.11.mlp.gate_proj.weight'])
    model.model.layers[11].mlp.down_proj.weight = nn.Parameter(data['model.layers.11.mlp.down_proj.weight'])
    model.model.layers[11].mlp.up_proj.weight = nn.Parameter(data['model.layers.11.mlp.up_proj.weight'])
    model.model.layers[11].input_layernorm.weight = nn.Parameter(data['model.layers.11.input_layernorm.weight'])
    model.model.layers[11].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.11.post_attention_layernorm.weight'])
    model.model.layers[12].self_attn.q_proj.weight = nn.Parameter(data['model.layers.12.self_attn.q_proj.weight'])
    model.model.layers[12].self_attn.k_proj.weight = nn.Parameter(data['model.layers.12.self_attn.k_proj.weight'])
    model.model.layers[12].self_attn.v_proj.weight = nn.Parameter(data['model.layers.12.self_attn.v_proj.weight'])
    model.model.layers[12].self_attn.o_proj.weight = nn.Parameter(data['model.layers.12.self_attn.o_proj.weight'])
    model.model.layers[12].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.12.self_attn.rotary_emb.inv_freq'])
    model.model.layers[12].mlp.gate_proj.weight = nn.Parameter(data['model.layers.12.mlp.gate_proj.weight'])
    model.model.layers[12].mlp.down_proj.weight = nn.Parameter(data['model.layers.12.mlp.down_proj.weight'])
    model.model.layers[12].mlp.up_proj.weight = nn.Parameter(data['model.layers.12.mlp.up_proj.weight'])
    model.model.layers[12].input_layernorm.weight = nn.Parameter(data['model.layers.12.input_layernorm.weight'])
    model.model.layers[12].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.12.post_attention_layernorm.weight'])
    model.model.layers[13].self_attn.q_proj.weight = nn.Parameter(data['model.layers.13.self_attn.q_proj.weight'])
    model.model.layers[13].self_attn.k_proj.weight = nn.Parameter(data['model.layers.13.self_attn.k_proj.weight'])
    model.model.layers[13].self_attn.v_proj.weight = nn.Parameter(data['model.layers.13.self_attn.v_proj.weight'])
    model.model.layers[13].self_attn.o_proj.weight = nn.Parameter(data['model.layers.13.self_attn.o_proj.weight'])
    model.model.layers[13].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.13.self_attn.rotary_emb.inv_freq'])
    model.model.layers[13].mlp.gate_proj.weight = nn.Parameter(data['model.layers.13.mlp.gate_proj.weight'])
    model.model.layers[13].mlp.down_proj.weight = nn.Parameter(data['model.layers.13.mlp.down_proj.weight'])
    model.model.layers[13].mlp.up_proj.weight = nn.Parameter(data['model.layers.13.mlp.up_proj.weight'])
    model.model.layers[13].input_layernorm.weight = nn.Parameter(data['model.layers.13.input_layernorm.weight'])
    model.model.layers[13].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.13.post_attention_layernorm.weight'])
    model.model.layers[14].self_attn.q_proj.weight = nn.Parameter(data['model.layers.14.self_attn.q_proj.weight'])
    model.model.layers[14].self_attn.k_proj.weight = nn.Parameter(data['model.layers.14.self_attn.k_proj.weight'])
    model.model.layers[14].self_attn.v_proj.weight = nn.Parameter(data['model.layers.14.self_attn.v_proj.weight'])
    model.model.layers[14].self_attn.o_proj.weight = nn.Parameter(data['model.layers.14.self_attn.o_proj.weight'])
    model.model.layers[14].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.14.self_attn.rotary_emb.inv_freq'])
    model.model.layers[14].mlp.gate_proj.weight = nn.Parameter(data['model.layers.14.mlp.gate_proj.weight'])
    model.model.layers[14].mlp.down_proj.weight = nn.Parameter(data['model.layers.14.mlp.down_proj.weight'])
    model.model.layers[14].mlp.up_proj.weight = nn.Parameter(data['model.layers.14.mlp.up_proj.weight'])
    model.model.layers[14].input_layernorm.weight = nn.Parameter(data['model.layers.14.input_layernorm.weight'])
    model.model.layers[14].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.14.post_attention_layernorm.weight'])
    model.model.layers[15].self_attn.q_proj.weight = nn.Parameter(data['model.layers.15.self_attn.q_proj.weight'])
    model.model.layers[15].self_attn.k_proj.weight = nn.Parameter(data['model.layers.15.self_attn.k_proj.weight'])
    model.model.layers[15].self_attn.v_proj.weight = nn.Parameter(data['model.layers.15.self_attn.v_proj.weight'])
    model.model.layers[15].self_attn.o_proj.weight = nn.Parameter(data['model.layers.15.self_attn.o_proj.weight'])
    model.model.layers[15].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.15.self_attn.rotary_emb.inv_freq'])
    model.model.layers[15].mlp.gate_proj.weight = nn.Parameter(data['model.layers.15.mlp.gate_proj.weight'])
    model.model.layers[15].mlp.down_proj.weight = nn.Parameter(data['model.layers.15.mlp.down_proj.weight'])
    model.model.layers[15].mlp.up_proj.weight = nn.Parameter(data['model.layers.15.mlp.up_proj.weight'])
    model.model.layers[15].input_layernorm.weight = nn.Parameter(data['model.layers.15.input_layernorm.weight'])
    model.model.layers[15].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.15.post_attention_layernorm.weight'])
    model.model.layers[16].self_attn.q_proj.weight = nn.Parameter(data['model.layers.16.self_attn.q_proj.weight'])
    model.model.layers[16].self_attn.k_proj.weight = nn.Parameter(data['model.layers.16.self_attn.k_proj.weight'])
    model.model.layers[16].self_attn.v_proj.weight = nn.Parameter(data['model.layers.16.self_attn.v_proj.weight'])
    model.model.layers[16].self_attn.o_proj.weight = nn.Parameter(data['model.layers.16.self_attn.o_proj.weight'])
    model.model.layers[16].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.16.self_attn.rotary_emb.inv_freq'])
    model.model.layers[16].mlp.gate_proj.weight = nn.Parameter(data['model.layers.16.mlp.gate_proj.weight'])
    model.model.layers[16].mlp.down_proj.weight = nn.Parameter(data['model.layers.16.mlp.down_proj.weight'])
    model.model.layers[16].mlp.up_proj.weight = nn.Parameter(data['model.layers.16.mlp.up_proj.weight'])
    model.model.layers[16].input_layernorm.weight = nn.Parameter(data['model.layers.16.input_layernorm.weight'])
    model.model.layers[16].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.16.post_attention_layernorm.weight'])
    model.model.layers[17].self_attn.q_proj.weight = nn.Parameter(data['model.layers.17.self_attn.q_proj.weight'])
    model.model.layers[17].self_attn.k_proj.weight = nn.Parameter(data['model.layers.17.self_attn.k_proj.weight'])
    model.model.layers[17].self_attn.v_proj.weight = nn.Parameter(data['model.layers.17.self_attn.v_proj.weight'])
    model.model.layers[17].self_attn.o_proj.weight = nn.Parameter(data['model.layers.17.self_attn.o_proj.weight'])
    model.model.layers[17].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.17.self_attn.rotary_emb.inv_freq'])
    model.model.layers[17].mlp.gate_proj.weight = nn.Parameter(data['model.layers.17.mlp.gate_proj.weight'])
    model.model.layers[17].mlp.down_proj.weight = nn.Parameter(data['model.layers.17.mlp.down_proj.weight'])
    model.model.layers[17].mlp.up_proj.weight = nn.Parameter(data['model.layers.17.mlp.up_proj.weight'])
    model.model.layers[17].input_layernorm.weight = nn.Parameter(data['model.layers.17.input_layernorm.weight'])
    model.model.layers[17].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.17.post_attention_layernorm.weight'])
    model.model.layers[18].self_attn.q_proj.weight = nn.Parameter(data['model.layers.18.self_attn.q_proj.weight'])
    model.model.layers[18].self_attn.k_proj.weight = nn.Parameter(data['model.layers.18.self_attn.k_proj.weight'])
    model.model.layers[18].self_attn.v_proj.weight = nn.Parameter(data['model.layers.18.self_attn.v_proj.weight'])
    model.model.layers[18].self_attn.o_proj.weight = nn.Parameter(data['model.layers.18.self_attn.o_proj.weight'])
    model.model.layers[18].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.18.self_attn.rotary_emb.inv_freq'])
    model.model.layers[18].mlp.gate_proj.weight = nn.Parameter(data['model.layers.18.mlp.gate_proj.weight'])
    model.model.layers[18].mlp.down_proj.weight = nn.Parameter(data['model.layers.18.mlp.down_proj.weight'])
    model.model.layers[18].mlp.up_proj.weight = nn.Parameter(data['model.layers.18.mlp.up_proj.weight'])
    model.model.layers[18].input_layernorm.weight = nn.Parameter(data['model.layers.18.input_layernorm.weight'])
    model.model.layers[18].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.18.post_attention_layernorm.weight'])
    model.model.layers[19].self_attn.q_proj.weight = nn.Parameter(data['model.layers.19.self_attn.q_proj.weight'])
    model.model.layers[19].self_attn.k_proj.weight = nn.Parameter(data['model.layers.19.self_attn.k_proj.weight'])
    model.model.layers[19].self_attn.v_proj.weight = nn.Parameter(data['model.layers.19.self_attn.v_proj.weight'])
    model.model.layers[19].self_attn.o_proj.weight = nn.Parameter(data['model.layers.19.self_attn.o_proj.weight'])
    model.model.layers[19].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.19.self_attn.rotary_emb.inv_freq'])
    model.model.layers[19].mlp.gate_proj.weight = nn.Parameter(data['model.layers.19.mlp.gate_proj.weight'])
    model.model.layers[19].mlp.down_proj.weight = nn.Parameter(data['model.layers.19.mlp.down_proj.weight'])
    model.model.layers[19].mlp.up_proj.weight = nn.Parameter(data['model.layers.19.mlp.up_proj.weight'])
    model.model.layers[19].input_layernorm.weight = nn.Parameter(data['model.layers.19.input_layernorm.weight'])
    model.model.layers[19].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.19.post_attention_layernorm.weight'])
    model.model.layers[20].self_attn.q_proj.weight = nn.Parameter(data['model.layers.20.self_attn.q_proj.weight'])
    model.model.layers[20].self_attn.k_proj.weight = nn.Parameter(data['model.layers.20.self_attn.k_proj.weight'])
    model.model.layers[20].self_attn.v_proj.weight = nn.Parameter(data['model.layers.20.self_attn.v_proj.weight'])
    model.model.layers[20].self_attn.o_proj.weight = nn.Parameter(data['model.layers.20.self_attn.o_proj.weight'])
    model.model.layers[20].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.20.self_attn.rotary_emb.inv_freq'])
    model.model.layers[20].mlp.gate_proj.weight = nn.Parameter(data['model.layers.20.mlp.gate_proj.weight'])
    model.model.layers[20].mlp.down_proj.weight = nn.Parameter(data['model.layers.20.mlp.down_proj.weight'])
    model.model.layers[20].mlp.up_proj.weight = nn.Parameter(data['model.layers.20.mlp.up_proj.weight'])
    model.model.layers[20].input_layernorm.weight = nn.Parameter(data['model.layers.20.input_layernorm.weight'])
    model.model.layers[20].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.20.post_attention_layernorm.weight'])
    model.model.layers[21].self_attn.q_proj.weight = nn.Parameter(data['model.layers.21.self_attn.q_proj.weight'])
    model.model.layers[21].self_attn.k_proj.weight = nn.Parameter(data['model.layers.21.self_attn.k_proj.weight'])
    model.model.layers[21].self_attn.v_proj.weight = nn.Parameter(data['model.layers.21.self_attn.v_proj.weight'])
    model.model.layers[21].self_attn.o_proj.weight = nn.Parameter(data['model.layers.21.self_attn.o_proj.weight'])
    model.model.layers[21].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.21.self_attn.rotary_emb.inv_freq'])
    model.model.layers[21].mlp.gate_proj.weight = nn.Parameter(data['model.layers.21.mlp.gate_proj.weight'])
    model.model.layers[21].mlp.down_proj.weight = nn.Parameter(data['model.layers.21.mlp.down_proj.weight'])
    model.model.layers[21].mlp.up_proj.weight = nn.Parameter(data['model.layers.21.mlp.up_proj.weight'])
    model.model.layers[21].input_layernorm.weight = nn.Parameter(data['model.layers.21.input_layernorm.weight'])
    model.model.layers[21].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.21.post_attention_layernorm.weight'])
    model.model.layers[22].self_attn.q_proj.weight = nn.Parameter(data['model.layers.22.self_attn.q_proj.weight'])
    model.model.layers[22].self_attn.k_proj.weight = nn.Parameter(data['model.layers.22.self_attn.k_proj.weight'])
    model.model.layers[22].self_attn.v_proj.weight = nn.Parameter(data['model.layers.22.self_attn.v_proj.weight'])
    model.model.layers[22].self_attn.o_proj.weight = nn.Parameter(data['model.layers.22.self_attn.o_proj.weight'])
    model.model.layers[22].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.22.self_attn.rotary_emb.inv_freq'])
    model.model.layers[22].mlp.gate_proj.weight = nn.Parameter(data['model.layers.22.mlp.gate_proj.weight'])
    model.model.layers[22].mlp.down_proj.weight = nn.Parameter(data['model.layers.22.mlp.down_proj.weight'])
    model.model.layers[22].mlp.up_proj.weight = nn.Parameter(data['model.layers.22.mlp.up_proj.weight'])
    model.model.layers[22].input_layernorm.weight = nn.Parameter(data['model.layers.22.input_layernorm.weight'])
    model.model.layers[22].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.22.post_attention_layernorm.weight'])
    model.model.layers[23].self_attn.q_proj.weight = nn.Parameter(data['model.layers.23.self_attn.q_proj.weight'])
    model.model.layers[23].self_attn.k_proj.weight = nn.Parameter(data['model.layers.23.self_attn.k_proj.weight'])
    model.model.layers[23].self_attn.v_proj.weight = nn.Parameter(data['model.layers.23.self_attn.v_proj.weight'])
    model.model.layers[23].self_attn.o_proj.weight = nn.Parameter(data['model.layers.23.self_attn.o_proj.weight'])
    model.model.layers[23].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.23.self_attn.rotary_emb.inv_freq'])
    model.model.layers[23].mlp.gate_proj.weight = nn.Parameter(data['model.layers.23.mlp.gate_proj.weight'])
    model.model.layers[23].mlp.down_proj.weight = nn.Parameter(data['model.layers.23.mlp.down_proj.weight'])
    model.model.layers[23].mlp.up_proj.weight = nn.Parameter(data['model.layers.23.mlp.up_proj.weight'])
    model.model.layers[23].input_layernorm.weight = nn.Parameter(data['model.layers.23.input_layernorm.weight'])
    model.model.layers[23].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.23.post_attention_layernorm.weight'])
    model.model.layers[24].self_attn.q_proj.weight = nn.Parameter(data['model.layers.24.self_attn.q_proj.weight'])
    model.model.layers[24].self_attn.k_proj.weight = nn.Parameter(data['model.layers.24.self_attn.k_proj.weight'])
    model.model.layers[24].self_attn.v_proj.weight = nn.Parameter(data['model.layers.24.self_attn.v_proj.weight'])
    model.model.layers[24].self_attn.o_proj.weight = nn.Parameter(data['model.layers.24.self_attn.o_proj.weight'])
    model.model.layers[24].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.24.self_attn.rotary_emb.inv_freq'])
    model.model.layers[24].mlp.gate_proj.weight = nn.Parameter(data['model.layers.24.mlp.gate_proj.weight'])
    model.model.layers[24].mlp.down_proj.weight = nn.Parameter(data['model.layers.24.mlp.down_proj.weight'])
    model.model.layers[24].mlp.up_proj.weight = nn.Parameter(data['model.layers.24.mlp.up_proj.weight'])
    model.model.layers[24].input_layernorm.weight = nn.Parameter(data['model.layers.24.input_layernorm.weight'])
    model.model.layers[24].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.24.post_attention_layernorm.weight'])
    model.model.layers[25].self_attn.q_proj.weight = nn.Parameter(data['model.layers.25.self_attn.q_proj.weight'])
    model.model.layers[25].self_attn.k_proj.weight = nn.Parameter(data['model.layers.25.self_attn.k_proj.weight'])
    model.model.layers[25].self_attn.v_proj.weight = nn.Parameter(data['model.layers.25.self_attn.v_proj.weight'])
    model.model.layers[25].self_attn.o_proj.weight = nn.Parameter(data['model.layers.25.self_attn.o_proj.weight'])
    model.model.layers[25].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.25.self_attn.rotary_emb.inv_freq'])
    model.model.layers[25].mlp.gate_proj.weight = nn.Parameter(data['model.layers.25.mlp.gate_proj.weight'])
    model.model.layers[25].mlp.down_proj.weight = nn.Parameter(data['model.layers.25.mlp.down_proj.weight'])
    model.model.layers[25].mlp.up_proj.weight = nn.Parameter(data['model.layers.25.mlp.up_proj.weight'])
    model.model.layers[25].input_layernorm.weight = nn.Parameter(data['model.layers.25.input_layernorm.weight'])
    model.model.layers[25].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.25.post_attention_layernorm.weight'])
    model.model.layers[26].self_attn.q_proj.weight = nn.Parameter(data['model.layers.26.self_attn.q_proj.weight'])
    model.model.layers[26].self_attn.k_proj.weight = nn.Parameter(data['model.layers.26.self_attn.k_proj.weight'])
    model.model.layers[26].self_attn.v_proj.weight = nn.Parameter(data['model.layers.26.self_attn.v_proj.weight'])
    model.model.layers[26].self_attn.o_proj.weight = nn.Parameter(data['model.layers.26.self_attn.o_proj.weight'])
    model.model.layers[26].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.26.self_attn.rotary_emb.inv_freq'])
    model.model.layers[26].mlp.gate_proj.weight = nn.Parameter(data['model.layers.26.mlp.gate_proj.weight'])
    model.model.layers[26].mlp.down_proj.weight = nn.Parameter(data['model.layers.26.mlp.down_proj.weight'])
    model.model.layers[26].mlp.up_proj.weight = nn.Parameter(data['model.layers.26.mlp.up_proj.weight'])
    model.model.layers[26].input_layernorm.weight = nn.Parameter(data['model.layers.26.input_layernorm.weight'])
    model.model.layers[26].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.26.post_attention_layernorm.weight'])
    model.model.layers[27].self_attn.q_proj.weight = nn.Parameter(data['model.layers.27.self_attn.q_proj.weight'])
    model.model.layers[27].self_attn.k_proj.weight = nn.Parameter(data['model.layers.27.self_attn.k_proj.weight'])
    model.model.layers[27].self_attn.v_proj.weight = nn.Parameter(data['model.layers.27.self_attn.v_proj.weight'])
    model.model.layers[27].self_attn.o_proj.weight = nn.Parameter(data['model.layers.27.self_attn.o_proj.weight'])
    model.model.layers[27].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.27.self_attn.rotary_emb.inv_freq'])
    model.model.layers[27].mlp.gate_proj.weight = nn.Parameter(data['model.layers.27.mlp.gate_proj.weight'])
    model.model.layers[27].mlp.down_proj.weight = nn.Parameter(data['model.layers.27.mlp.down_proj.weight'])
    model.model.layers[27].mlp.up_proj.weight = nn.Parameter(data['model.layers.27.mlp.up_proj.weight'])
    model.model.layers[27].input_layernorm.weight = nn.Parameter(data['model.layers.27.input_layernorm.weight'])
    model.model.layers[27].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.27.post_attention_layernorm.weight'])
    model.model.layers[28].self_attn.q_proj.weight = nn.Parameter(data['model.layers.28.self_attn.q_proj.weight'])
    model.model.layers[28].self_attn.k_proj.weight = nn.Parameter(data['model.layers.28.self_attn.k_proj.weight'])
    model.model.layers[28].self_attn.v_proj.weight = nn.Parameter(data['model.layers.28.self_attn.v_proj.weight'])
    model.model.layers[28].self_attn.o_proj.weight = nn.Parameter(data['model.layers.28.self_attn.o_proj.weight'])
    model.model.layers[28].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.28.self_attn.rotary_emb.inv_freq'])
    model.model.layers[28].mlp.gate_proj.weight = nn.Parameter(data['model.layers.28.mlp.gate_proj.weight'])
    model.model.layers[28].mlp.down_proj.weight = nn.Parameter(data['model.layers.28.mlp.down_proj.weight'])
    model.model.layers[28].mlp.up_proj.weight = nn.Parameter(data['model.layers.28.mlp.up_proj.weight'])
    model.model.layers[28].input_layernorm.weight = nn.Parameter(data['model.layers.28.input_layernorm.weight'])
    model.model.layers[28].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.28.post_attention_layernorm.weight'])
    model.model.layers[29].self_attn.q_proj.weight = nn.Parameter(data['model.layers.29.self_attn.q_proj.weight'])
    model.model.layers[29].self_attn.k_proj.weight = nn.Parameter(data['model.layers.29.self_attn.k_proj.weight'])
    model.model.layers[29].self_attn.v_proj.weight = nn.Parameter(data['model.layers.29.self_attn.v_proj.weight'])
    model.model.layers[29].self_attn.o_proj.weight = nn.Parameter(data['model.layers.29.self_attn.o_proj.weight'])
    model.model.layers[29].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.29.self_attn.rotary_emb.inv_freq'])
    model.model.layers[29].mlp.gate_proj.weight = nn.Parameter(data['model.layers.29.mlp.gate_proj.weight'])
    model.model.layers[29].mlp.down_proj.weight = nn.Parameter(data['model.layers.29.mlp.down_proj.weight'])
    model.model.layers[29].mlp.up_proj.weight = nn.Parameter(data['model.layers.29.mlp.up_proj.weight'])
    model.model.layers[29].input_layernorm.weight = nn.Parameter(data['model.layers.29.input_layernorm.weight'])
    model.model.layers[29].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.29.post_attention_layernorm.weight'])
    model.model.layers[30].self_attn.q_proj.weight = nn.Parameter(data['model.layers.30.self_attn.q_proj.weight'])
    model.model.layers[30].self_attn.k_proj.weight = nn.Parameter(data['model.layers.30.self_attn.k_proj.weight'])
    model.model.layers[30].self_attn.v_proj.weight = nn.Parameter(data['model.layers.30.self_attn.v_proj.weight'])
    model.model.layers[30].self_attn.o_proj.weight = nn.Parameter(data['model.layers.30.self_attn.o_proj.weight'])
    model.model.layers[30].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.30.self_attn.rotary_emb.inv_freq'])
    model.model.layers[30].mlp.gate_proj.weight = nn.Parameter(data['model.layers.30.mlp.gate_proj.weight'])
    model.model.layers[30].mlp.down_proj.weight = nn.Parameter(data['model.layers.30.mlp.down_proj.weight'])
    model.model.layers[30].mlp.up_proj.weight = nn.Parameter(data['model.layers.30.mlp.up_proj.weight'])
    model.model.layers[30].input_layernorm.weight = nn.Parameter(data['model.layers.30.input_layernorm.weight'])
    model.model.layers[30].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.30.post_attention_layernorm.weight'])
    model.model.layers[31].self_attn.q_proj.weight = nn.Parameter(data['model.layers.31.self_attn.q_proj.weight'])
    model.model.layers[31].self_attn.k_proj.weight = nn.Parameter(data['model.layers.31.self_attn.k_proj.weight'])
    model.model.layers[31].self_attn.v_proj.weight = nn.Parameter(data['model.layers.31.self_attn.v_proj.weight'])
    model.model.layers[31].self_attn.o_proj.weight = nn.Parameter(data['model.layers.31.self_attn.o_proj.weight'])
    model.model.layers[31].self_attn.rotary_emb.inv_freq = nn.Parameter(
        data['model.layers.31.self_attn.rotary_emb.inv_freq'])
    model.model.layers[31].mlp.gate_proj.weight = nn.Parameter(data['model.layers.31.mlp.gate_proj.weight'])
    model.model.layers[31].mlp.down_proj.weight = nn.Parameter(data['model.layers.31.mlp.down_proj.weight'])
    model.model.layers[31].mlp.up_proj.weight = nn.Parameter(data['model.layers.31.mlp.up_proj.weight'])
    model.model.layers[31].input_layernorm.weight = nn.Parameter(data['model.layers.31.input_layernorm.weight'])
    model.model.layers[31].post_attention_layernorm.weight = nn.Parameter(
        data['model.layers.31.post_attention_layernorm.weight'])
    model.model.norm.weight = nn.Parameter(data['model.norm.weight'])
    model.lm_head.weight = nn.Parameter(data['lm_head.weight'])
else:
    model = load_lgm_weights(model, data, True)
