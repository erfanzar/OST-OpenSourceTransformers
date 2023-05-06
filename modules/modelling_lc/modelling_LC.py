import math

import torch
from torch import nn
from einops import rearrange
from typing import Optional, Union


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


class PMSNorm(nn.Module):
    def __init__(self, config):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps: Optional[float] = config.eps

    def pms(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

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


class MultiheadAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self, hidden_size: int, number_of_attention_heads: int, clip_qkv: Optional[float] = None,
                 softmax_scale: Optional[float] = None,
                 device: Optional[str] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.number_of_attention_heads = number_of_attention_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size // self.number_of_attention_heads)

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, device=device)

        self.q_ln = nn.LayerNorm(self.hidden_size, device=device)
        self.k_ln = nn.LayerNorm(self.hidden_size, device=device)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device)

    def forward(self, x, attn_bias=None, attention_mask=None):
        qkv = self.qkv(x)

        (query, key, value) = qkv.chunk(3, dim=2)
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        attn_weights = scale_dot_production(query, key, value, self.number_of_attention_heads, bias=attn_bias,
                                            softmax_scale=self.softmax_scale)
        return self.out_proj(attn_weights)
