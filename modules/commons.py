import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1377)

__all__ = ['BLM']


class BLM(nn.Module):
    def __init__(self, vocab_size):
        super(BLM, self).__init__()

        self.f = F
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        tokens = self.token_embedding(idx)
        if targets is not None:
            B, T, C = tokens.shape
            tokens = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(tokens, targets)
        else:
            loss = None
        return tokens, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            token, loss = self(idx)
            # print(f'token size = {token.shape}')
            token = token[:, -1, :]
            probs = F.softmax(token, dim=-1)
            # print(f'prob shape = {probs.shape}')
            idx_next = torch.multinomial(probs, num_samples=1)
            # print(f'idx before = {idx} | idx_next = {idx_next}')
            idx = torch.cat([idx, idx_next], 1)
            # print(f'idx after = {idx}')
            # print('-' * 10)
        return idx
