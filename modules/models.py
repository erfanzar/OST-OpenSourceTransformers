import logging

from typing import Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from erutils.lightning import build_alibi_tensor

from .cross_modules import LLmPConfig
from .modeling_LLmP import LLmPBlock, PMSNorm
from .modeling_PGT import PGTConfig, PGTBlock, Adafactor

logger = logging.getLogger(__name__)

__all__ = ['PGT', 'LLmP', 'LLmPBlock', 'LLmPConfig', 'Adafactor',
           'PGTConfig']


class PGT(nn.Module):
    def __init__(self, config: PGTConfig):
        super(PGT, self).__init__()
        self.config: PGTConfig = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.ln = PMSNorm(config)
        self.transformer = nn.ModuleList(
            [PGTBlock(config, layer_idx_1=i) for i in range(config.n_layers)]
        )

        self.fc = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self,
                input_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden = self.ln(self.wte(input_ids))
        if head_mask is None:
            head_mask = [None] * self.config.n_layers
        assert len(head_mask) == len(self.transformer)
        for i, (block, h_m) in enumerate(zip(self.transformer, head_mask)):
            logger.debug(f'BEFORE BLOCK NUMBER {i} hidden : {hidden.shape} STATUS [UNKNOWN]')
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                heads_mask=h_m
            )
            logger.debug(f'BEFORE BLOCK NUMBER {i} hidden : {hidden.shape} STATUS [PASS]')
        hidden = self.fc(self.ln(hidden))
        loss = None
        if labels is not None:
            shift_r = hidden[..., -1, :].contiguous()
            shift_l = labels[..., 1:, :].contiguous()
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(shift_r.view(-1, shift_r.size(-1)), shift_l.view(-1))
        return hidden, loss

    @torch.no_grad()
    def generate(self, idx, generate=5000, temp=1, eos: int = 2, attention_mask=None):
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
        for _ in range(generate):
            idx = idx[:, -self.config.max_sentence_length:]
            pred, _ = self.forward(idx, attention_mask=attention_mask)
            pred = pred[:, -1, :] / temp
            pred = F.softmax(pred, dim=-1)
            next_index = torch.multinomial(pred, 1)
            idx = torch.cat([idx, next_index], 1)
            if next_index[0] == eos:
                break
            else:
                yield next_index


class LLmP(nn.Module):
    def __init__(self, config: LLmPConfig):
        super(LLmP, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wte_ln = PMSNorm(config)
        self.h = nn.ModuleList([LLmPBlock(config=config, layer_index=i) for i in range(config.n_layers)])
        self.ln = PMSNorm(config)

        self.out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.freq = precompute_frq_cis(config.hidden_size // config.n_heads, config.max_sentence_length * 2).to(
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
        logger.debug(
            f'We Got INPUT ---**--- :  [ input _ids : {input_ids.shape}] [ attention _mask : {attention_mask.shape if attention_mask is not None else None} ]')
        # self.freq = self.freq.to(input_ids.device)
        # chosen_freq = self.freq[:seq_len]
        # logger.debug(f'chosen_freq : {chosen_freq.shape}')
        attention_mask = attention_mask.to(input_ids.device)
        alibi = build_alibi_tensor(attention_mask=attention_mask.view(attention_mask.size()[0], -1),
                                   dtype=attention_mask.dtype,
                                   n_heads=self.config.n_heads).to(input_ids.device)

        x = self.wte_ln(self.wte(input_ids))
        logger.debug(f'word tokenizing shape ==> : {x.shape}')
        for i, h in enumerate(self.h):
            logger.debug(f'At Block Index  : \033[32m{i}\033[92m')
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
                                                     (0, self.config.max_sentence_length - tokens.size(-1)),
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
