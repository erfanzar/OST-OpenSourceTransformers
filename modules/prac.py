import torch
from torch import nn
from dataclasses import dataclass
import logging
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)


class Config(object):
    eps: Optional[float] = 1e-5
    hidden_size: Optional[int] = 512
    alternative = Optional[int] = 4
    residual_norm = Optional[float] = 0.1
    is_gated = Optional[bool] = True
    is_decoder = Optional[bool] = False


class T5Attention(nn.Module):
    def __init__(self, config: Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):

        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):

        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):

            if key_value_states is None:

                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:

                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:

                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:

                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class PMSNorm(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(PMSNorm, self).__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def norm(self, x: Optional[torch.Tensor]):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Optional[torch.Tensor]):
        x = self.norm(x)
        return self.weight * x


class DenseActDense(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(DenseActDense, self).__init__()
        self.drop = nn.Dropout(config.residual_norm)
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size * config.alternative, bias=False)
        self.wo = nn.Linear(config.hidden_size * config.alternative, config.hidden_size, bias=False)
        self.act = torch.nn.functional.gelu

    def forward(self, x: Optional[torch.Tensor]):
        x = self.drop(self.act(self.w1(x)))
        return self.wo(x)


class DenseGatedActDense(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(DenseGatedActDense, self).__init__()
        self.drop = nn.Dropout(config.residual_norm)
        self.ln = PMSNorm(config=config)
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size * config.alternative, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size * config.alternative, bias=False)
        self.wo = nn.Linear(config.hidden_size * config.alternative, config.hidden_size, bias=False)
        self.act = torch.nn.functional.gelu

    def forward(self, x: Optional[torch.Tensor]):
        x = self.drop(self.w2(self.act(self.w1(x))))
        if x.dtype != self.wo.weight.dtype and self.wo.weight.dtype != torch.int8:
            x = x.to(self.wo.weight.dtype)
        return self.wo(x)


class LayerFF(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(LayerFF, self).__init__()
        layer = nn.ModuleList()
        layer.append(PMSNorm(config))
        layer.append(DenseGatedActDense(config) if config.is_gated else DenseActDense(config))
        self.drop = nn.Dropout(config.residual_norm)
        self.layer = layer

    def forward(self, x: Optional[torch.Tensor]):
        f = x
        for m in self.layer:
            f = m(x)
        x = x + self.drop(f)
        return x


class Block(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(Block, self).__init__()


class Stock(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(Stock, self).__init__()


class LLmPU(nn.Module):
    def __init__(self, config: Optional[Config]):
        super(LLmPU, self).__init__()
