import math
# LT Added
import torch
from torch import nn
from einops import rearrange
from typing import Optional, Union, List, Tuple, Type
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
import triton
from triton import language as tl
from triton.ops import matmul


class LtConfig(PretrainedConfig):
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
                 softmax_scale: float = None,
                 alibi_bias_max: int = 8
                 ):
        super().__init__(eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id)
        self.max_sequence_length = max_sequence_length
        self.weight_decay = weight_decay
        self.alibi_bias_max = alibi_bias_max
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
    m = torch.arange(1, number_of_attention_heads + 1, dtype=torch.float32).to(device)
    m = m.mul(alibi_bias_max / closest_power_2)
    slope = 1 / torch.pow(2, m)
    if closest_power_2 != number_of_attention_heads:
        slope = torch.cat([slope[1::2], slope[::2]], dim=-1)[:number_of_attention_heads]
    return slope.view(1, number_of_attention_heads, 1, 1)


def build_alibi_bias(max_length, number_of_attention_heads, device, dtype, alibi_bias_max=8):
    t = torch.arange(1 - max_length, 1).reshape(1, 1, 1, max_length).to(device)
    slopes = gen_slopes(number_of_attention_heads=number_of_attention_heads, alibi_bias_max=alibi_bias_max,
                        device=device)
    t = t * slopes
    return t.to(dtype)


def build_attention_bias(attention_bias: torch.Tensor, max_length: int, alibi_bias_max: int, num_attention_heads: int,
                         dtype,
                         device):
    attention_bias = attention_bias.add(build_alibi_bias(max_length=max_length, alibi_bias_max=alibi_bias_max,
                                                         number_of_attention_heads=num_attention_heads, dtype=dtype,
                                                         device=device))
    return attention_bias


class LtNorm(nn.Module):
    def __init__(self, config: LtConfig):
        super(LtNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    @staticmethod
    def pms(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        x = self.pms(x.float())
        return x * self.weight


def scale_dot_production(
        q, k, v, attention_head: int, bias=None, softmax_scale: float = None
):
    q = rearrange(q, 'b s (h d) -> b h s d', h=attention_head)
    k = rearrange(k, 'b s (h d) -> b h d s', h=attention_head)
    v = rearrange(v, 'b s (h d) -> b h s d', h=attention_head)
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


@triton.jit
def scale_dot_production_triton(
        q, k, v, attention_head: int, bias=None, softmax_scale: float = None
):
    q = rearrange(q, 'b s (h d) -> b h s d', h=attention_head)
    k = rearrange(k, 'b s (h d) -> b h d s', h=attention_head)
    v = rearrange(v, 'b s (h d) -> b h s d', h=attention_head)
    min_val = torch.finfo(q.dtype).min
    s_q, s_k = q.size(-2), k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(q.size(-1))

    attn_weight = matmul(q, k) * softmax_scale
    if bias is not None:
        attn_weight += bias
    s = max(s_q, s_k)
    causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
    causal_mask = causal_mask.tril()
    causal_mask = causal_mask.to(torch.bool)
    causal_mask = ~causal_mask
    causal_mask = causal_mask[-s_q:, -s_k:]
    attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = tl.softmax(attn_weight)
    out = matmul(attn_weight, v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    return out


class LTAttention(nn.Module):
    def __init__(self, config: LtConfig):
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

    def forward(self, x, attention_bias=None):
        qkv = self.qkv(x)

        (query, key, value) = qkv.chunk(3, dim=2)
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)
        if attention_bias is not None:
            attention_bias = attention_bias[:, :, -query.size(1):, -key.size(1):]
        attn_weights = scale_dot_production_triton(query, key, value, self.num_attention_heads, bias=attention_bias,
                                                   softmax_scale=self.softmax_scale)
        return self.out_proj(attn_weights)


class LtMLP(nn.Module):
    def __init__(self, config: LtConfig):
        super().__init__()
        # self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU('none')

    def forward(self, x):
        # return self.down(self.act(self.gate(x)) * self.up(x))
        return self.down(self.act(self.gate(x)))


class LtBlock(nn.Module):
    def __init__(self, config: LtConfig):
        super().__init__()
        self.ln = LtNorm(config)
        self.self_attn = LTAttention(config)
        self.mlp = LtMLP(config)

    def forward(self, x, attention_bias, attention_mask=None):
        x = self.self_attn(x,
                           attention_bias=attention_bias,
                           attention_mask=attention_mask
                           ) + x
        residual = x
        x = self.ln(x)
        return self.mlp(x) + residual


class LtPreTrainedModel(PreTrainedModel):
    config_class = LtConfig
    base_model_prefix = 'lt'


class LtModel(LtPreTrainedModel):
    def __init__(self, config: LtConfig):
        super().__init__(config=config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ln = LtNorm(config)
        self.blocks = nn.ModuleList([LtBlock(config) for _ in range(config.num_hidden_layers)])
        self.cfg = config
        self.attn_bias_shape = (1, config.num_attention_heads, 1, config.max_sequence_length)
        self.is_bias_initialized = False
        self.alibi_bias_max = config.alibi_bias_max

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, value: nn.Module):
        self.wte = value

    def is_fsdp_wrap_block(self, module) -> bool:
        return isinstance(module, LtBlock)

    def get_fsdp_wrap_block(self) -> Type[nn.Module]:
        return LtBlock

    @torch.no_grad()
    def _build_attention_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor] = None):
        if not self.is_bias_initialized:
            if self.attn_bias_shape:
                self.attention_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
                self.attention_bias = build_attention_bias(attention_bias=self.attention_bias,
                                                           num_attention_heads=self.cfg.num_attention_heads,
                                                           max_length=self.cfg.max_sequence_length,
                                                           alibi_bias_max=self.alibi_bias_max,
                                                           device=device, dtype=dtype, )
            self.is_bias_initialized = True
        if self.attention_bias is not None:
            self.attention_bias = self.attention_bias.to(dtype=dtype, device=device)
        attention_bias = self.attention_bias

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attention_bias is None:
                attention_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                attention_bias = attention_bias[:, :, :, -s_k:]
            min_val = torch.finfo(attention_bias.dtype).min
            attention_bias = attention_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return attention_bias

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.ByteTensor] = None,
                use_attn_bias: Optional[bool] = True,
                output_attentions: Optional[bool] = False,
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
                return_dict: Optional[bool] = None):
        return_dict = return_dict or self.config.return_dict
        s = input_ids.size(1)
        if past_key_values is not None:
            return NotImplementedError('past_key_values is not supported yet')
        if output_attentions:
            return NotImplementedError('output_attention is not supported yet')

        assert s > self.config.max_sequence_length, 'max_sequence_length should be higher than input length'
        hidden = self.wte(input_ids)
        attention_bias, _ = self._build_attention_bias(device=hidden.device,
                                                       attention_mask=attention_mask,
                                                       dtype=hidden.dtype)
        for block in self.blocks:
            hidden = block(
                hidden, attention_bias
            )

        hidden = self.ln(hidden)
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden,
                hidden_states=None
            )
        else:
            return hidden, None


class LtModelForCausalLM(LtPreTrainedModel):
    def __init__(self, config: LtConfig):
        super().__init__(config=config)
        self.model = LtModel(config=config)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.wte

    def set_input_embeddings(self, value: nn.Module):
        self.model.wte = value

    def is_fsdp_wrap_block(self, module) -> bool:
        return isinstance(module, LtBlock)

    def get_fsdp_wrap_block(self) -> Type[nn.Module]:
        return LtBlock

    def get_model(self):
        return self.model

    def set_model(self, value):
        self.model = value

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.ByteTensor] = None,
                use_attn_bias: Optional[bool] = True,
                output_attentions: Optional[bool] = False,
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: Optional[bool] = None):
        out = self.model(input_ids, attention_mask, use_attn_bias, output_attentions, past_key_values, return_dict)
        hidden_sate = out.last_hidden_state if return_dict else out[0]
        logit = torch.nn.functional.linear(hidden_sate, self.model.wte.weight)
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = torch.nn.functional.cross_entropy(logit.view(-1, logit.size(-1)), labels.to(logit.device).view(-1))
        if return_dict:
            return CausalLMOutput(logits=logit, loss=loss)
        else:
            return logit, loss
