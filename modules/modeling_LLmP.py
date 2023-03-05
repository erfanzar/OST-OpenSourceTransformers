from .cross_modules import PMSNorm, FeedForward, Attention, LLmPConfig
import torch
from torch import nn
from typing import Optional, List, Union, Tuple


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
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.block = Attention(config=config, layer_index=layer_index)
        self.ln1 = PMSNorm(config)
        self.ln2 = PMSNorm(config)
        self.config: LLmPConfig = config
        self.ffd = FeedForward(config)

    def forward(self, hidden: Optional[torch.Tensor], freq: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        residual = self.ln1(hidden)
        hidden = hidden + self.block(residual, freq=freq, attention_mask=attention_mask)
        residual = self.ln2(hidden)
        hidden = hidden + self.ffd(residual)
        return hidden

    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            pad_id: int,
            eos_id: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ) -> List[int]:
        batch_size = len(prompts)
        params = self.config
        assert batch_size <= self.config.max_batch_size, (batch_size, self.config.max_batch_size)

        prompt_tokens = prompts

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_sentence_length, max_gen_len + max_prompt_size)

        tokens = torch.full((batch_size, total_len), pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            inp = tokens[:, -self.config.max_sentence_length]
            attention_mask = (inp != 0).float()
            logits = self.forward(inp, attention_mask=attention_mask)
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)

        for i, t in enumerate(tokens.tolist()):

            t = t[: len(prompt_tokens[i]) + max_gen_len]

            try:
                t = t[: t.index(eos_id)]
            except ValueError:
                pass

        return t
