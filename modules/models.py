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

__all__ = ['PGTForCausalLM', 'LLmP', 'LLmPBlock', 'LLmPConfig', 'Adafactor',
           'PGTConfig']


class PGT(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PGTBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask=None
    ) -> Union[Tuple]:

        batch_size = input_ids.size(0)
        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)

            attention_mask = attention_mask[:, None, None, :]

            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embed_in(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,

                head_mask=head_mask[i],
            )

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class PGTForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = PGT(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None
    ) -> Union[Tuple]:
        hidden_states = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,

        )

        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        output = (lm_logits,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output


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
