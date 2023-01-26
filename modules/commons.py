from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# torch.manual_seed(1377)

__all__ = ['Head', 'BLM']


class Head(nn.Module):
    def __init__(self, c: int = 32, head_size: int = 16, chunk: int = 8):
        super(Head, self).__init__()
        self.head_size = head_size
        self.key = nn.Linear(c, head_size, bias=False)
        self.query = nn.Linear(c, head_size, bias=False)
        self.value = nn.Linear(c, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(chunk, chunk)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


class BLM(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int = 8, n_embedded: int = 328, head_size: int = 16, n_head: int = 6,
                 chunk_size: int = 256

                 ):

        super(BLM, self).__init__()

        self.vocab_size = vocab_size
        self.chunk = chunk_size
        self.head_size = head_size

        self.token_embedding = nn.Embedding(vocab_size, n_embedded)
        self.position_embedding = nn.Embedding(chunk_size, n_embedded)

        self.blocks = nn.Sequential(
            *[Block(chunk_size=chunk_size, n_embedded=n_embedded, n_head=n_head) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(n_embedded)  # final layer norm
        self.lm_head = nn.Linear(n_embedded, vocab_size)

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = pos_emb + tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            tokens = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(tokens, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.chunk:]
            # print(idx_cond)
            token, loss = self(idx_cond)
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


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embedded: int = 328):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedded, 4 * n_embedded),
            nn.ReLU(),
            nn.Linear(4 * n_embedded, n_embedded),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num, head_size: int = 16, chunk_size: int = 8, n_embedded: int = 328):
        super(MultiHeadAttention, self).__init__()

        self.m = nn.ModuleList([Head(head_size=head_size, chunk=chunk_size, c=n_embedded) for _ in range(num)])
        self.proj = nn.Linear(n_embedded, n_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.m], dim=-1)
        x = self.dp(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, n_head, n_embedded: int = 328, chunk_size: int = 8):
        super(Block, self).__init__()
        head_size = n_embedded // n_head
        self.sa = MultiHeadAttention(n_head, head_size=head_size, chunk_size=chunk_size, n_embedded=n_embedded)
        self.ffwd = FeedForward(n_embedded=n_embedded)
        self.ln1 = nn.LayerNorm(n_embedded)
        self.ln2 = nn.LayerNorm(n_embedded)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
