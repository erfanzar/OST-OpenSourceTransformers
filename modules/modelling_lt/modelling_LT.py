import math

import torch
from torch import nn
from einops import rearrange
from typing import Optional, Union
from transformers import PreTrainedModel, PretrainedConfig


class LTConfig(PretrainedConfig):
    def __init__(self,
                 initializer_range: float = 0.02,
                 hidden_size: int = 2048,
                 bos_token_id=2,
                 eos_token_id=1,
                 pad_token_id=0,
                 intermediate_size: int = 8192,
                 num_hidden_layers: int = 16,
                 vocab_size: int = 32000,
                 num_attention_heads: int = 16,
                 weight_decay: float = 0.02,
                 max_sequence_length: int = 1536,
                 softmax_scale: float = None
                 ):
        super().__init__(eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id)
        self.max_sequence_length = max_sequence_length
        self.weight_decay = weight_decay
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.softmax_scale = softmax_scale


def gen_slopes(number_of_attention_heads, alibi_bias_max=8, device=None):
    closest_power_2 = 2 ** math.ceil(math.log2(number_of_attention_heads))
    m = torch.arange(1, number_of_attention_heads + 1).to(device)
    m = m.mul(alibi_bias_max / number_of_attention_heads)
    slope = 1 / math.pow(2, m)
    if closest_power_2 != number_of_attention_heads:
        slope = torch.cat([slope[1::2], slope[::2]], dim=-1)[:number_of_attention_heads]
    return slope.view(1, number_of_attention_heads, 1, 1)


def build_alibi_bias(max_length, number_of_attention_heads, alibi_bias_max=8):
    t = torch.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    slopes = gen_slopes(number_of_attention_heads=number_of_attention_heads, alibi_bias_max=alibi_bias_max)
    t = t * slopes
    return t


class LtNorm(nn.Module):
    def __init__(self, config: LTConfig):
        super(LtNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    @staticmethod
    def pms(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        x = self.pms(x.float())
        return x * self.weight


def scale_dot_production(
        q, k, v, attention_head: int, bias=None, softmax_scale: float = None,
):
    q = rearrange(q, 'b s (h d) -> b s h d', h=attention_head)
    k = rearrange(k, 'b s (h d) -> b s d h', h=attention_head)
    v = rearrange(v, 'b s (h d) -> b s h d', h=attention_head)
    min_val = torch.finfo(q.dtype).min
    s_q, s_k = q.size(-2), k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(q.size(-1))

    attn_weight = (q @ k) * softmax_scale
    if bias is not None:
        attn_weight += bias
    s = max(s_q, s_k)
    causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
    causal_mask = causal_mask.tril()
    causal_mask = causal_mask.to(torch.bool)
    causal_mask = ~causal_mask
    causal_mask = causal_mask[-s_q:, -s_k:]
    attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, -1)
    out = attn_weight @ v
    out = rearrange(out, 'b h s d -> b s (h d)')
    return out


class LTAttention(nn.Module):
    def __init__(self, config: LTConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.softmax_scale = config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size // self.num_attention_heads)

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.q_ln = LtNorm(config)
        self.k_ln = LtNorm(config)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, attn_bias=None, attention_mask=None):
        qkv = self.qkv(x)

        (query, key, value) = qkv.chunk(3, dim=2)
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        attn_weights = scale_dot_production(query, key, value, self.num_attention_heads, bias=attn_bias,
                                            softmax_scale=self.softmax_scale)
        return self.out_proj(attn_weights)


class LtMLP(nn.Module):
    def __init__(self, config: LTConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU('none')

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class LtBlock(nn.Module):
    def __init__(self, config: LTConfig):
        super().__init__()
        self.ln = LtNorm(config)
        self.self_attn = LTAttention(config)
        self.mlp = LtMLP(config)

    def forward(self, x, attn_bias, attention_mask=None):
        x = self.self_attn(x,
                           attn_bias=attn_bias,
                           attention_mask=attention_mask
                           ) + x
        residual = x
        x = self.ln(x)
        return self.mlp(x) + residual
