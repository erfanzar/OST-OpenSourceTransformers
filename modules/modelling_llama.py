import logging
import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import torch
from erutils.loggers import show_hyper_parameters
from torch import nn
from transformers import GPT2Tokenizer

from .dataset import Tokens

logger = logging.getLogger(__name__)


class Tokenizer(Tokens):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=self.sos, eos_token=self.eos,
                                                       pad_token=self.pad)


@dataclass
class LLamaConfig:
    eps: Optional[float] = 1e-6
    hidden_size: Optional[int] = 1200
    n_heads: Optional[int] = 12
    n_layers: Optional[int] = 14
    vocab_size: Optional[int] = None
    max_sentence_length: Optional[int] = 512
    max_batch_size: Optional[int] = 32
    lr: Optional[float] = 3e-4
    weight_decay: Optional[float] = 2e-1
    epochs: Optional[int] = 100
    device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu'


class PMSNorm(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(PMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps: Optional[float] = config.eps

    def pms(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        x = self.pms(x.float())
        return x * self.weight


def precompute_frq_cis(dim: int, end: int, theta: float = 10000.0) -> Optional[torch.Tensor]:
    rng = torch.arange(0, dim, 2)
    freq = 1.0 / (theta ** (rng[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freq.device)  # type: ignore
    freq = torch.outer(t, freq).float()  # type: ignore
    freq = torch.polar(torch.ones_like(freq), freq)  # complex64
    return freq


def broadcast_shaping(x: Optional[torch.Tensor], frq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    ndim = x.ndim
    assert ndim > 1
    assert frq.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return frq.view(*shape)


def rotary_embedding(xq: Optional[torch.Tensor], xk: Optional[torch.Tensor],
                     frq: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    frq = broadcast_shaping(xq_, frq)
    xq_out = torch.view_as_real(xq_ * frq).flatten(3)
    xk_out = torch.view_as_real(xk_ * frq).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.wo = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self.wo(nn.functional.silu(self.w1(x)) * self.w2(x))


class LLamaAttention(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(LLamaAttention, self).__init__()
        self.local_rank = config.n_heads // 1
        self.head_dim = config.hidden_size // config.n_heads
        assert config.hidden_size % config.n_heads == 0
        self.wq = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                            )
        self.wk = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                            )
        self.wv = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False,
                            )
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False,
                            )
        # self.cash_k = nn.Parameter(torch.zeros(
        #     (config.max_batch_size, config.max_sentence_length, self.local_rank, self.head_dim)).to(config.device),
        #                            requires_grad=False)
        # self.cash_v = nn.Parameter(torch.zeros(
        #     (config.max_batch_size, config.max_sentence_length, self.local_rank, self.head_dim)).to(config.device),
        #                            requires_grad=False
        #                            )
    def forward(self, x: Optional[torch.Tensor], pos_start: int, freq: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        batch_, seq_len_, _ = x.shape
        xq = self.wq(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xv = self.wv(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        xk = self.wk(x).view(batch_, seq_len_, self.local_rank, self.head_dim)
        logger.debug(f'xq : {xq.shape} \nxv : {xv.shape}\nxk : {xk.shape}')
        # using rotary embedding for key and query
        if freq is not None:
            xq, xk = rotary_embedding(xq=xq, xk=xk, frq=freq)
        else:
            logger.debug('Freq is None')
        # we need to cash key and values

        # self.cash_k[:batch_, pos_start:pos_start + seq_len_] = xk
        # self.cash_v[:batch_, pos_start:pos_start + seq_len_] = xv
        # key = self.cash_k[:batch_, pos_start:pos_start + seq_len_]
        # value = self.cash_v[:batch_, pos_start:pos_start + seq_len_]
        key = xk
        value = xv
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
            _, _, s, h = attention.shape
            attention += mask[:, :, :s, :h]
        attention = nn.functional.softmax(attention, dim=-1)
        # after matmul [batch, num_heads, seq_len , head_dim]
        comb = torch.matmul(attention, value).permute(0, 2, 1, 3).contiguous().view(batch_, seq_len_, -1)
        return self.wo(comb)


class LLamaBlock(nn.Module):
    def __init__(self, config: LLamaConfig, layer_id: int):
        super(LLamaBlock, self).__init__()
        self.layer_id: int = layer_id
        self.ln1 = PMSNorm(config)
        self.ln2 = PMSNorm(config)
        self.attention = LLamaAttention(config)
        self.ffw = FeedForward(config)

    def forward(self, x: Optional[torch.Tensor], pos_start: int,
                mask: Optional[torch.Tensor], freq: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        h = x + self.attention(self.ln1(x), mask=mask, pos_start=pos_start, freq=freq)
        return h + self.ffw(self.ln2(h))


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LLamaModel(nn.Module):
    def __init__(self, config: LLamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLamaBlock(config, layer_id) for layer_id in range(config.n_layers)])
        self.norm = PMSNorm(config)
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.freq = precompute_frq_cis(config.hidden_size // config.n_heads, config.max_sentence_length * 2)

    def forward(self, tokens: torch.Tensor, pos_start: int):
        _batch, seq_len = tokens.shape
        h = self.wte(tokens)
        mask = None
        self.freq = self.freq.to(h.device)
        chosen_freq = self.freq[pos_start:pos_start + seq_len]
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=pos_start + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, pos_start=pos_start, mask=mask, freq=chosen_freq)
        h = self.norm(h)
        # output = self.output(h[:, -1, :])  # only compute last logits
        output = self.output(h)
        return output

    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ) -> List[str]:
        batch_size = len(prompts)
        params = self.model.params
        assert batch_size <= self.config.max_batch_size, (batch_size, self.config.max_batch_size)

        prompt_tokens = prompts

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            t = t[: len(prompt_tokens[i]) + max_gen_len]

            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


if __name__ == "__main__":
    config = LLamaConfig()
    config.vocab_size = 5027
    config.hidden_size = 1400
    config.n_heads = 20
    config.n_layers = 10
    model = LLamaModel(config=config)
    show_hyper_parameters(config)
    print('Model Initialized with ', sum(v.numel() for v in model.parameters()) / 1e6, ' Million Parameters')
