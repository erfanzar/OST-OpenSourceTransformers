from typing import Optional

import torch
from torch import nn

from modules.cross_modules import PMSNorm, FeedForward, Attention
from dataclasses import dataclass
from typing import Union, Tuple, Iterable
from erutils.lightning import build_alibi_tensor
from transformers import PretrainedConfig


@dataclass
class LLmPConfig(PretrainedConfig):
    eps: Optional[float] = 1e-5
    hidden_size: Optional[int] = 1200
    use_layer_index_scaling: Optional[bool] = False
    n_heads: Optional[int] = 12
    n_layers: Optional[int] = 14
    vocab_size: Optional[int] = None
    lr: Optional[float] = 3e-4
    weight_decay: Optional[float] = 2e-1
    epochs: Optional[int] = 100
    hidden_dropout: Optional[float] = 0.1
    embed_dropout: Optional[float] = 0.1
    device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)

    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LLmPBlock(nn.Module):
    def __init__(self, config: Optional[LLmPConfig], layer_index: Optional[int] = None):
        super(LLmPBlock, self).__init__()
        self.block = Attention(config=config, layer_index=layer_index)
        self.ln1 = PMSNorm(config)
        self.ln2 = PMSNorm(config)
        self.config: LLmPConfig = config
        self.ffd = FeedForward(config)

    def forward(self, hidden: Optional[torch.Tensor], alibi: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        residual = self.ln1(hidden)
        hidden = hidden + self.block(residual, alibi=alibi, attention_mask=attention_mask)
        residual = self.ln2(hidden)
        hidden = hidden + self.ffd(residual)
        return hidden


class LLmP(nn.Module):
    def __init__(self, config: LLmPConfig):
        super(LLmP, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wte_ln = PMSNorm(config)
        self.h = nn.ModuleList([LLmPBlock(config=config, layer_index=i) for i in range(config.n_layers)])
        self.ln = PMSNorm(config)

        self.out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.freq = precompute_frq_cis(config.hidden_size // config.n_heads, config.max_sequence_length * 2).to(
        #     self.dtype)
        # i dont use freq or rotaty embedding in LLmP anymore
        self.config = config
        # self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.002)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.002)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor],
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        batch, seq_len = input_ids.shape
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
            # attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = torch.ones(input_ids.shape).to(torch.float32)
            # attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]

        # self.freq = self.freq.to(input_ids.device)
        # chosen_freq = self.freq[:seq_len]
        # logger.debug(f'chosen_freq : {chosen_freq.shape}')
        attention_mask = attention_mask.to(input_ids.device)
        alibi = build_alibi_tensor(attention_mask=attention_mask.view(attention_mask.size()[0], -1),
                                   dtype=attention_mask.dtype,
                                   n_heads=self.config.n_heads).to(input_ids.device)

        x = self.wte_ln(self.wte(input_ids))

        for i, h in enumerate(self.h):
            x = h(x, attention_mask=attention_mask, alibi=alibi)
        logits = self.out(self.ln(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

    def generate(
            self,
            tokens: Optional[torch.Tensor],
            eos_id: int,
            pad_id: int,
            attention_mask=None,
            max_gen_len: int = 20,
            temperature: float = 0.9,
            top_p: float = 0.95,
    ) -> Iterable[torch.Tensor]:
        def sample_top_p(probs, p):
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            _next_token = torch.multinomial(probs_sort, num_samples=1)

            _next_token = torch.gather(probs_idx, -1, _next_token)
            return _next_token

        if attention_mask is True:
            attention_mask = torch.nn.functional.pad((tokens != 0).float(),
                                                     (0, self.config.max_sequence_length - tokens.size(-1)),
                                                     value=pad_id)
        # attention_mask = None
        for i in range(max_gen_len):
            # tokens = tokens[:, :]
            logits, _ = self.forward(tokens, attention_mask)
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(*tokens.shape[:-1], 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.view(-1)[0] != eos_id:

                yield next_token.view(1, -1)
            else:
                break
