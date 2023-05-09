import math

from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from einops import rearrange, einsum
import torch
from typing import Type
from collections import namedtuple
from erutils import make2d
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base=512, use_pos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_pos = use_pos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not self.use_pos:
            return freqs, torch.ones(1, device=device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    depth = x.shape[-1]
    x1 = x[..., :depth]
    x2 = x[..., depth:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(position, tensor, scale=1.0):
    return (tensor * position.cos() * scale) + (rotate_half(tensor) * position.sin() * scale)


# feedforward

class SwiGLU(nn.Module):
    @staticmethod
    def forward(x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class PalmConfig(PretrainedConfig):
    def __init__(self,
                 hidden_size: int = 2048,
                 bos_token_id: int = 2,
                 eos_token_id: int = 1,
                 pad_token_id: int = 0,
                 num_attention_heads: int = 8,
                 dim_attention_head: int = 128,
                 intermediate_size: int = 8192,
                 num_hidden_layers: int = 16,
                 vocab_size: int = 50304,
                 std_wte: float = 0.02,
                 flash_attn: bool = False,
                 **kwargs
                 ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
        self.hidden_size = hidden_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_attention_heads = num_attention_heads
        self.dim_attention_head = dim_attention_head
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.std_wte = std_wte
        self.flash_attn = flash_attn
        self.__dict__.update(**kwargs)


Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class Attention(nn.Module):
    def __init__(
            self,
            causal=False,
            use_flash_attn=False
    ):
        super().__init__()

        self.causal = causal
        self.register_buffer("attention_mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not use_flash_attn:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if self.attention_mask is not None and self.attention_mask.shape[-1] >= n:
            return self.attention_mask[:n, :n]

        attention_mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("attention_mask", attention_mask, persistent=False)
        return attention_mask

    def flash_attn(self, q, k, v, attention_mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda
        k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, 'b j -> b 1 1 j')
            attention_mask = attention_mask.expand(-1, heads, q_len, -1)

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=self.causal
            )

        return out

    def forward(self, q, k, v, attention_mask=None, return_score=False):
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        if self.use_flash_attn:
            return self.flash_attn(q, k, v, attention_mask=attention_mask)
        spa = einsum("b h i d, b j d -> b h i j", q, k) * scale
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, 'b j -> b 1 1 j')
            spa = spa.masked_fill(~attention_mask, -torch.finfo(spa.dtype).max)
        if self.causal:
            causal_mask = self.get_mask(n, device)
            spa = spa.masked_fill(causal_mask, -torch.finfo(spa.dtype).max)
        attn = spa.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        return out if return_score is False else out, attn


class PalmBlock(nn.Module):
    def __init__(self, config: PalmConfig):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

        inner_dim = config.dim_attention_head * config.num_attention_heads
        self.split_dim = (
            inner_dim, config.dim_attention_head, config.dim_attention_head, (config.intermediate_size * 2))

        self.heads = config.num_attention_heads
        self.scale = 1 / math.sqrt(config.dim_attention_head)
        self.rotary_emb = RotaryEmbedding(config.dim_attention_head)
        self.attend = Attention(
            causal=True,
            use_flash_attn=config.flash_attn
        )
        self.fused_attn_ff_proj = nn.Linear(config.hidden_size, sum(self.split_dim), bias=False)
        self.attn_out = nn.Linear(config.dim_attention_head * config.num_attention_heads, config.hidden_size,
                                  bias=False)
        self.o_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_o_proj = SwiGLU()

        self.register_buffer("pos_emb", None, persistent=False)

    def forward(self, hidden_state, attention_mask=None, return_attention_score=False):
        n, device, h = hidden_state.shape[1], hidden_state.device, self.heads

        hidden_state = self.norm(hidden_state)

        q, k, v, ff = self.fused_attn_ff_proj(hidden_state).split(self.fused_dims, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        positions, scale = self.get_rotary_embedding(n, device)

        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)
        score = None
        if return_attention_score:
            out, score = self.attend.forward(q, k, v, attention_mask=attention_mask,
                                             return_score=return_attention_score)
        else:
            out = self.attend.forward(q, k, v, attention_mask=attention_mask, return_score=return_attention_score)
        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff)

        return attn_out + ff_out, score


class PalmPretrainedModule(PreTrainedModel):
    base_model_prefix = 'model'
    config_class = PalmConfig
    supports_gradient_checkpointing = True


class PalmModel(PalmPretrainedModule):
    def __init__(self, config: PalmConfig):
        super().__init__(config=config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PalmBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln = LayerNorm(config.hidden_size)
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None, return_dict=False):
        hidden = self.wte(input_ids)
        score = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def comp_1(module):
                    def comp_2(*inputs):
                        return module.forward(*inputs)

                    return comp_2

                hidden, score[i] = torch.utils.checkpoint.checkpoint(
                    comp_1(layer),
                    hidden,
                    attention_mask
                )
            else:
                hidden, score[i] = layer.forward(hidden, attention_mask=attention_mask)

        if return_dict:
            return BaseModelOutput(
                last_hidden_state=hidden,
                hidden_states=score
            )
        else:
            return self.ln(hidden)


class PalmForCausalLM(PalmPretrainedModule):
    def __init__(self, config: PalmConfig):
        super().__init__(config=config)
        self.transformer = PalmModel(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        torch.nn.init.normal_(self.transformer.wte.weight, std=config.std_wte)

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Module):
        self.transformer.wte = value

    def is_fsdp_wrap_block(self, module) -> bool:
        return isinstance(module, PalmBlock)

    def get_fsdp_wrap_block(self) -> Type[nn.Module]:
        return PalmBlock

    def get_model(self):
        return self.transformer

    def set_model(self, value):
        self.transformer = value

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=False, **kwargs):
        tr = self.transformer.forward(input_ids, attention_mask=attention_mask, return_dict=return_dict)
        out = self.lm_head(tr)
        loss = None
        if labels is not None:
            shifted_logits = out[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(make2d(shifted_logits), shifted_labels.view(-1))
        if return_dict:
            return CausalLMOutput(
                logits=out,
                loss=loss,
                attentions=tr
            )
        else:
            return loss, out if loss else out

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {'input_ids': input_ids, 'attention_mask': attention_mask, **kwargs}
