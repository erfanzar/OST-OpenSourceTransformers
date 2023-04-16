import logging
from typing import Dict, Tuple, Any
from typing import Optional

import torch
import math
from torch import nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class PGTConfig:
    dtype: Optional[torch.dtype] = torch.float32
    hidden_size: Optional[int] = 2048
    eps: Optional[float] = 1e-5
    n_heads: Optional[int] = 8
    n_layers: Optional[int] = 12
    epochs: Optional[int] = 100
    scale_attn_by_layer_idx: Optional[bool] = True
    vocab_size: Optional[int] = -1
    max_sequence_length: Optional[int] = 512
    hidden_dropout: Optional[float] = 0.1
    intermediate_size: Optional[int] = 4
    residual_dropout: Optional[float] = 0.1
    training: Optional[bool] = True
    attention_dropout: Optional[float] = 0.1
    weight_decay: Optional[float] = 2e-1
    initializer_range: Optional[float] = 0.02
    lr: Optional[float] = 3e-4
    rotary_pct: Optional[float] = 0.25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_parallel_residual: Optional[bool] = True
    silu: Optional[bool] = False


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PMSNorm(nn.Module):
    def __init__(self, config: PGTConfig, eps: Optional[float] = 1e-5):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = eps

    def norm(self, x: Optional[torch.Tensor]):
        nrm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return nrm

    def forward(self, x):
        x = self.norm(x)
        return x * self.weight


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class PGTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor = 0,
            head_mask: Optional[torch.FloatTensor] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,

    ):
        has_layer_past = layer_past is not None

        qkv = self.query_key_value(hidden_states)

        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size: 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        return attn_output

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):

        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)

        tensor = tensor.view(new_shape).permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):

        tensor = tensor.permute(0, 2, 1, 3).contiguous()

        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)

        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):

        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min

        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class PGTFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class PGTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = PGTAttention(config)
        self.mlp = PGTFeedForward(config)

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor],
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,

    ):
        attn_output = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        if self.use_parallel_residual:

            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:

            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        return hidden_states


Eps2 = Tuple[float, float]
ParamGroup = Dict[str, Any]


class Adafactor(Optimizer):

    def __init__(
            self,
            params,
            lr=None,
            eps2: Eps2 = (1e-30, 1e-3),
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
            beta1=None,
            weight_decay: float = 0.0,
            scale_parameter: bool = True,
            relative_step: bool = True,
            warmup_init: bool = False,
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    def _get_lr(self, param_group: ParamGroup, param_state) -> float:
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = (
                1e-6 * param_state['step']
                if param_group['warmup_init']
                else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps2'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(
            self, param_group: ParamGroup, param_shape: Tuple[int, ...]
    ) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(
            self,
            exp_avg_sq_row: torch.Tensor,
            exp_avg_sq_col: torch.Tensor,
            output: torch.Tensor,
    ) -> None:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adafactor does not support sparse gradients.'
                    )

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape
                )
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad_shape[:-1]
                        ).type_as(grad)
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0

                state['step'] += 1
                state['RMS'] = self._rms(p.data)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad ** 2) + group['eps2'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square
                    # of gradient
                    self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col, update
                    )
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(
                    max(1.0, self._rms(update) / group['clip_threshold'])
                )
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(
                        update, alpha=1 - group['beta1']
                    )
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                p.data.add_(-update)

        return loss
