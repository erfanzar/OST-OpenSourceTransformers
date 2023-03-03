import math

import torch
from torch import nn
from typing import Optional, Union
from dataclasses import dataclass
from transformers import GPT2Tokenizer
from dataset import Tokens
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
from fairscale.nn.model_parallel import ColumnParallelLinear, RowParallelLinear
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenizer(Tokens):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=self.sos, eos_token=self.eos,
                                                       pad_token=self.pad)


@dataclass
class LLamaConfig:
    eps: Optional[float] = 1e-6
    hidden_size: Optional[int] = 680
    n_heads: Optional[int] = 12
    n_layers: Optional[int] = 8
    vocab_size: Optional[int] = None
    max_sentence_length: Optional[int] = 512
    max_batch_size: Optional[int] = 32
    device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu'


class PMSNorm(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps: Optional[float] = config.eps

    def pms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self.pms(x.float()).type_as(x)
        return x * self.weight


def broadcast_shaping(x: Optional[torch.Tensor], frq: Optional[torch.Tensor]):
    ndim = x.ndim
    assert ndim > 1
    assert frq.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return frq.view(*shape)


def rotary_embedding(xq: Optional[torch.Tensor], xk: Optional[torch.Tensor], frq: Optional[torch.Tensor]):
    xq_ = torch.view_as_complex(xq.float().view(xq.shape[-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(xk.shape[-1], -1, 2))
    frq = broadcast_shaping(xq_, frq)
    xq_out = torch.view_as_real(xq_ * frq).flatten(3)
    xk_out = torch.view_as_real(xk_ * frq).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LLamaAttention(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(LLamaAttention, self).__init__()
        self.local_rank = config.n_heads // get_model_parallel_rank()
        self.head_dim = config.hidden_size // config.n_heads
        self.wq = ColumnParallelLinear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                                       gather_output=False, init_method=lambda x: x)
        self.wk = ColumnParallelLinear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                                       gather_output=False, init_method=lambda x: x)
        self.wv = ColumnParallelLinear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                                       gather_output=False, init_method=lambda x: x)
        self.wo = RowParallelLinear(config.n_heads * self.head_dim, config.hidden_size, bias=False,
                                    input_is_parallel=True, init_method=lambda x: x)
        self.cash_k = torch.zeros(
            (config.max_batch_size, config.max_sentence_length, self.local_rank, self.head_dim)).to(config.device)
        self.cash_v = torch.zeros(
            (config.max_batch_size, config.max_sentence_length, self.local_rank, self.head_dim)).to(config.device)

    def forward(self, x: Optional[torch.Tensor], pos_start: int, frq: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None):
        batch_, seq_len_, _ = x.shape
        xq = self.wq(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xv = self.wv(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xk = self.wk(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        logger.debug(f'xq : {xq.shape} \nxv : {xv.shape}\nxk : {xk.shape}')
        # using rotary embedding for key and query
        xq, xk = rotary_embedding(xq=xq, xk=xk, frq=frq)
        # we need to cash key and values
        self.cash_v = self.cash_v.to(xv)
        self.cash_k = self.cash_k.to(xk)
        self.cash_k[:batch_, pos_start:pos_start + seq_len_] = xk
        self.cash_v[:batch_, pos_start:pos_start + seq_len_] = xq
        key = self.cash_k[:batch_, pos_start:pos_start + seq_len_]
        value = self.cash_v[:batch_, pos_start:pos_start + seq_len_]
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        key = key.permute(0, 2, 1, 3)
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        value = value.permute(0, 2, 1, 3)
        # [batch, seq_len , num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query = xq.permute(0, 2, 1, 3)
        logger.debug(f'key : {key.shape} \nvalue : {value.shape}\nquery : {query.shape}')
        # key : [batch, num_heads, seq_len, head_dim] -> [batch, seq_len , num_heads, head_dim]
        # score : [batch, num_heads, seq_len , head_dim]
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logger.debug(f'score : {attention.shape}')
        if mask is not None:
            attention += mask
        attention = nn.functional.softmax(attention, dim=-1)
        # after matmul [batch, num_heads, seq_len , head_dim]
        comb = torch.matmul(attention, value).permute(0, 2, 1, 3).contiguous().view(batch_, seq_len_, -1)
        return self.wo(comb)


if __name__ == "__main__":
    v = Tokenizer()
    print('SUCCESS')
