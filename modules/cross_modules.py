import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
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
    dtype: Optional[torch.dtype] = torch.float32
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
    def __init__(self, config, eps: Optional[float] = 1e-5):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size, dtype=config.dtype))
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
        self.hidden_size = config.hidden_size
        self.wq = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.wk = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.wv = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False, dtype=config.dtype)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: Optional[torch.Tensor], alibi: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        batch_, seq_len_, _ = x.shape
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, head_dim, seq_len]
        query = self.wq(x).view(batch_, seq_len_, self.local_rank, self.head_dim).permute(0, 2, 1, 3).view(
            batch_ * self.local_rank, seq_len_, self.head_dim
        )
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        value = self.wv(x).view(batch_, seq_len_, self.local_rank, self.head_dim).permute(0, 2, 1, 3).view(
            batch_ * self.local_rank, seq_len_, self.head_dim
        )
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        key = self.wk(x).view(batch_, seq_len_, self.local_rank, self.head_dim).permute(0, 2, 3, 1).view(
            batch_ * self.local_rank, self.head_dim, seq_len_)
        _, _, key_len_ = key.shape

        attention = alibi.baddbmm(batch1=query, batch2=key, beta=1, alpha=math.sqrt(self.head_dim)).view(batch_,
                                                                                                         self.local_rank,
                                                                                                         seq_len_,
                                                                                                         key_len_)
        if self.use_layer_index_scaling:
            attention /= (self.layer_index + 1)
        logger.debug(f'attention : {attention.shape}')
        logger.debug(f'attention mask : {attention_mask.shape if attention_mask is not None else None}')
        _, _, s, h = attention.shape
        if attention_mask is not None:
            attention += attention_mask[:, :, :, :h]
        attention = nn.functional.softmax(attention, dim=-1)
        attention = self.drop(attention).view(batch_ * self.local_rank, seq_len_, key_len_)
        comb = torch.bmm(attention, value).view(batch_, -1, self.hidden_size)
        return self.wo(comb)


class FeedForward(nn.Module):
    def __init__(self, config, up: Optional[int] = 4):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size * up, bias=False, dtype=config.dtype)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size * up, bias=False, dtype=config.dtype)
        self.wo = nn.Linear(config.hidden_size * up, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self.wo(nn.functional.gelu(self.w1(x)) * self.w2(x))
