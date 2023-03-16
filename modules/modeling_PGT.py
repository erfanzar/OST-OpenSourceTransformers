import logging
from typing import Dict, Tuple, Any
from typing import Optional

import torch
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
    vocab_size: Optional[int] = -1
    max_sentence_length: Optional[int] = 512
    hidden_dropout: Optional[float] = 0.1
    training: Optional[bool] = True
    attention_dropout: Optional[float] = 0.1
    weight_decay: Optional[float] = 2e-1
    initializer_range: Optional[float] = 0.02
    lr: Optional[float] = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class PGTAttention(nn.Module):
    def __init__(self, config: PGTConfig, layer_idx=None):
        super(PGTAttention, self).__init__()
        self.layer_idx = layer_idx
        self.embedding = config.num_embedding
        self.n_heads = config.n_heads
        self.num_div = self.embedding // self.n_heads
        self.scale_attn_by_layer_idx = config.scale_attn_by_layer_idx
        assert self.embedding % self.n_heads != 0
        self.c_attn = nn.Linear(self.embedding, self.embedding * 3, bias=False)
        self.c_proj = nn.Linear(self.embedding, self.embedding, bias=False)
        self.residual_norm = PMSNorm(config)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.register_buffer('bias', torch.tril(
            torch.ones(config.max_sentence_length, config.max_sentence_length, dtype=torch.uint8,
                       device=config.device).view(1, 1,
                                                  config.max_sentence_length,
                                                  config.max_sentence_length)))
        lowest = torch.finfo(torch.float32).min
        self.register_buffer('masked_bias', lowest)

    def _split_heads(self, tensor: Optional[torch.Tensor]):
        new_shape = tensor.size()[:-1] + (self.n_heads, self.num_div)

        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor: Optional[torch.Tensor]):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.n_heads * self.num_div,)
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
        query, key, value = self.c_attn(hidden_state).split(self.embedding, dim=len(hidden_state.shape) - 1)
        query = self._split_heads(query).permute(0, 2, 1, 3)
        key = self._split_heads(key).permute(0, 2, 3, 1)
        value = self._split_heads(value).permute(0, 2, 1, 3)
        attn_output = self.residual_norm(
            self._attn(query=query, key=key, value=value, attention_mask=attention_mask, head_mask=head_mask))
        attn_output = (self.c_proj(self._merge_heads(attn_output)))
        return attn_output


class PGTFeedForward(nn.Module):
    def __init__(self, config: PGTConfig):
        super(PGTFeedForward, self).__init__()
        self.c_op = nn.Linear(config.num_embedding, config.num_embedding * config.intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.num_embedding * config.intermediate_size, config.num_embedding, bias=False)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.act = nn.functional.silu

    def forward(self, hidden_state):
        hidden_state = self.c_op(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.c_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class PGTBlock(nn.Module):
    def __init__(self, config: PGTConfig, layer_idx_1=None):
        super(PGTBlock, self).__init__()

        self.ln = PMSNorm(config)
        self.h = PGTAttention(config=config, layer_idx=layer_idx_1)
        self.mlp = PGTFeedForward(config)

    def forward(self,
                hidden_state: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.FloatTensor] = None,
                heads_mask: Optional[torch.FloatTensor] = None):
        residual_normed = self.ln(hidden_state)
        attn = self.h(residual_normed, attention_mask=attention_mask, heads_mask=heads_mask) + residual_normed
        hidden_state = self.mlp(attn)
        return hidden_state


Eps2 = Tuple[float, float]
ParamGroup = Dict[str, Any]


class Adafactor(Optimizer):

    def __init__(
            self,
            params: Params,
            lr: OptFloat = None,
            eps2: Eps2 = (1e-30, 1e-3),
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
            beta1: OptFloat = None,
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

    def _get_lr(self, param_group: ParamGroup, param_state: State) -> float:
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

    def step(self, closure: OptLossClosure = None) -> OptFloat:
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
