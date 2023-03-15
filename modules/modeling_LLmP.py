from typing import Optional

import torch
from torch import nn

from .cross_modules import PMSNorm, FeedForward, Attention, LLmPConfig


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
