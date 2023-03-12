import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from erutils.lightning import rotary_embedding
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class LLmPConfig:
    eps: Optional[float] = 1e-6
    hidden_size: Optional[int] = 1200
    use_layer_index_scaling: Optional[bool] = True
    n_heads: Optional[int] = 12
    n_layers: Optional[int] = 14
    vocab_size: Optional[int] = None
    max_sentence_length: Optional[int] = 512
    max_batch_size: Optional[int] = 32
    lr: Optional[float] = 3e-4
    weight_decay: Optional[float] = 2e-1
    epochs: Optional[int] = 100
    hidden_dropout: Optional[float] = 0.1
    embed_dropout: Optional[float] = 0.1
    device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu'


def precompute_frq_cis(dim: int, end: int, theta: float = 10000.0):
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freq.device)  # type: ignore
    freq = torch.outer(t, freq).float()  # type: ignore
    freq = torch.polar(torch.ones_like(freq), freq)  # complex64
    return freq


class PMSNorm(nn.Module):
    def __init__(self, config, eps: Optional[float] = 1e-6):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = eps

    def norm(self, x: Optional[torch.Tensor]):
        nrm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return nrm

    def forward(self, x):
        x = self.norm(x)
        return x * self.weight


class Attention(nn.Module):
    def __init__(self, config: Optional[LLmPConfig], layer_index: Optional[int] = None):
        super(Attention, self).__init__()
        self.layer_index = layer_index
        self.local_rank = config.n_heads
        self.use_layer_index_scaling = config.use_layer_index_scaling
        self.head_dim = config.hidden_size // config.n_heads
        assert config.hidden_size % config.n_heads == 0
        self.wq = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.register_buffer('bias', torch.triu(
            torch.full((1, 1, config.max_sentence_length, config.max_sentence_length), float('-inf')),
            diagonal=1))

    def forward(self, x: Optional[torch.Tensor], freq: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        batch_, seq_len_, _ = x.shape
        xq = self.wq(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xv = self.wv(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xk = self.wk(x).view(batch_, seq_len_, self.local_rank, self.head_dim)

        # using rotary embedding for key and query
        if freq is not None:
            xq, xk = rotary_embedding(xq, xk, freq=freq)
        else:
            logger.debug('Freq is None')
        # we need to cash key and values
        key = xk
        value = xv
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        key = key.permute(0, 2, 1, 3)
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        value = value.permute(0, 2, 1, 3)
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query = xq.permute(0, 2, 1, 3)
        # logger.debug(f'key : {key.shape} \nvalue : {value.shape}\nquery : {query.shape}')
        # key : [batch, num_heads, seq_len, head_dim] -> [batch, seq_len , num_heads, head_dim]
        # score : [batch, num_heads, seq_len , head_dim]
        attention = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(self.head_dim))
        if self.use_layer_index_scaling:
            attention /= (self.layer_index + 1)
        logger.debug(f'attention : {attention.shape}')
        logger.debug(f'attention mask : {attention_mask.shape if attention_mask is not None else None}')
        _, _, s, h = attention.shape

        if attention_mask is not None:
            attention += attention_mask[:, :, :, :h]
        attention += self.bias[:, :, :s, :h]
        attention = nn.functional.softmax(attention, dim=-1)
        # after matmul [batch, num_heads, seq_len , head_dim]
        comb = torch.matmul(attention, value).permute(0, 2, 1, 3).contiguous().view(batch_, seq_len_, -1)
        return self.wo(comb)


class FeedForward(nn.Module):
    def __init__(self, config, up: Optional[int] = 4):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size * up, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size * up, bias=False)
        self.wo = nn.Linear(config.hidden_size * up, config.hidden_size, bias=False)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self.wo(nn.functional.silu(self.w1(x)) * self.w2(x))
