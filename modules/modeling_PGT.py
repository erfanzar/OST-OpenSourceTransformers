import logging
from typing import Dict, Tuple, Any
from typing import Optional

import torch
import math
from torch import nn
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl

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
    max_sentence_length: Optional[int] = 512
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


class RotaryEmbedding(pl.LightningModule):
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
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PMSNorm(pl.LightningModule):
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


class PGTAttention(pl.LightningModule):
    def __init__(self, config: PGTConfig, layer_idx=None):
        super(PGTAttention, self).__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.head_size = self.hidden_size // self.n_heads
        self.scale_attn_by_layer_idx = config.scale_attn_by_layer_idx
        assert self.hidden_size // self.n_heads != 0
        self.c_attn = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.residual_norm = PMSNorm(config)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_dims = int(self.head_size * config.rotary_pct)
        self.register_buffer('bias', torch.tril(
            torch.ones(config.max_sentence_length, config.max_sentence_length, dtype=torch.uint8,
                       device=config.device).view(1, 1,
                                                  config.max_sentence_length,
                                                  config.max_sentence_length)))
        self.rope = RotaryEmbedding(self.rotary_dims, config.max_sentence_length)
        self.register_buffer('masked_bias', torch.tensor(float(-1e5)))

    def _split_heads(self, tensor: Optional[torch.Tensor]):
        new_shape = tensor.size()[:-1] + (self.n_heads, self.head_size)

        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor: Optional[torch.Tensor]):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.n_heads * self.head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask, head_mask):
        attn_weight = torch.matmul(query, key)

        attn_weight = attn_weight / torch.full([], value.size(-1) ** 0.5, dtype=attn_weight.dtype,
                                               device=attn_weight.device)
        if self.scale_attn_by_layer_idx:
            attn_weight /= self.layer_idx + 1

        key_len, query_len = key.size(-2), query.size(-2)
        masked = self.bias[:, :, key_len - query_len:query_len, :key_len].to(attn_weight.device)
        attn_weight = attn_weight.masked_fill(masked == 0, self.masked_bias)
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weight = attn_weight + attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn_weight = attn_weight.type(value.dtype)
        if head_mask is not None:
            attn_weight = attn_weight * head_mask

        attn_weight = torch.matmul(attn_weight, value)
        return attn_weight

    def forward(self, hidden_state: Optional[torch.Tensor], attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_state).split(self.hidden_size, dim=len(hidden_state.shape) - 1)
        query = self._split_heads(query).permute(0, 2, 1, 3)
        key = self._split_heads(key).permute(0, 2, 1, 3)
        value = self._split_heads(value).permute(0, 2, 1, 3)

        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        seq_len = key.shape[-2]
        offset = 0

        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        attn_output = self.residual_norm(
            self._attn(query=query, key=key, value=value, attention_mask=attention_mask, head_mask=head_mask))
        attn_output = (self.c_proj(self._merge_heads(attn_output)))
        return attn_output


class PGTFeedForward(pl.LightningModule):
    def __init__(self, config: PGTConfig):
        super(PGTFeedForward, self).__init__()
        self.c_op = nn.Linear(config.hidden_size, config.hidden_size * config.intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.hidden_size * config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.act = nn.functional.silu if config.silu else nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.c_op(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.c_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class PGTBlock(pl.LightningModule):
    def __init__(self, config: PGTConfig, layer_idx_1=None):
        super(PGTBlock, self).__init__()

        self.ln = PMSNorm(config)
        self.post_ln = PMSNorm(config)
        self.h = PGTAttention(config=config, layer_idx=layer_idx_1)
        self.mlp = PGTFeedForward(config)
        self.use_parallel_residual = config.use_parallel_residual

    def forward(self,
                hidden_state: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.FloatTensor] = None,
                heads_mask: Optional[torch.FloatTensor] = None):

        attn = self.h(self.ln(hidden_state), attention_mask=attention_mask, heads_mask=heads_mask)
        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            out = hidden_state + attn + self.mlp(self.post_ln(hidden_state))
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            out = attn + self.mlp(self.post_ln(attn))

        return out


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
