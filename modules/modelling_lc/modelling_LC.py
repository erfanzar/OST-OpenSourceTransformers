import math

import torch
from torch import nn
from einops import rearrange
from typing import Optional, Union


def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    closest_power_2 = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, n_heads + 1).to(device)
    m = m.mul(alibi_bias_max / n_heads)
    slope = 1 / math.pow(2, m)
    if closest_power_2 != n_heads:
        slope = torch.cat([slope[1::2], slope[::2]], dim=-1)[:n_heads]
    return slope.view(1, n_heads, 1, 1)


def build_alibi_bias(max_length, n_heads, alibi_bias_max=8):
    t = torch.arange(1 - max_length, 1).reshape(1, 1, 1, max_length)
    slopes = gen_slopes(n_heads=n_heads, alibi_bias_max=alibi_bias_max)
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

    def __init__(self, d_model: int, n_heads: int, attn_impl: str = 'triton', clip_qkv: Optional[float] = None,
                 qk_ln: bool = False, softmax_scale: Optional[float] = None, attn_pdrop: float = 0.0,
                 low_precision_layernorm: bool = False, device: Optional[str] = None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.n_heads)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)
        if self.attn_impl == 'torch':
            self.attn_fn = scale_dot_production

        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, attn_bias=None, attention_mask=None):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.chunk(3, dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        attn_weights = self.attn_fn(query, key, value, self.n_heads, bias=attn_bias, softmax_scale=self.softmax_scale)
        return self.out_proj(attn_weights)
