import typing

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config:
    chunk = 8
    head_size = 128
    n_embedded = 128
    num_layer = 6
    num_head = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = 0.2


class Head(nn.Module):
    def __init__(self, n_embedded: int, head_size: int, chunk: int):
        super(Head, self).__init__()

        self.key = nn.Linear(n_embedded, head_size, bias=False)
        self.query = nn.Linear(n_embedded, head_size, bias=False)
        self.value = nn.Linear(n_embedded, head_size, bias=False)

        self.register_buffer('bias', torch.tril(torch.ones(chunk, chunk)))

    def forward(self, k: torch.Tensor, q: torch.Tensor = None, v: torch.Tensor = None):
        if q is not None and v is not None:
            assert k.shape == q.shape and q.shape == v.shape
            b, t, c = k.shape
            key = self.key(k)
            query = self.query(q)
            value = self.value(v)

            attn = query @ key.transpose(-2, -1) * c ** -0.5
            attn = attn.masked_fill(self.bias[:t, :t] == 0, float('-inf'))
            attn = nn.functional.softmax(attn, dim=-1)
            wei = attn @ value
            return value
        else:
            x = k
            b, t, c = x.shape
            key = self.model.key(x)
            query = self.model.key(x)
            value = self.model.key(x)

            attn = query @ key.transpose(-2, -1) * c ** -0.5
            attn = attn.masked_fill(self.bias[:t, :t] == 0, float('-inf'))
            attn = nn.functional.softmax(attn, dim=-1)
            wei = attn @ value
            return value


class FeedForward(nn.Module):
    def __init__(self, n_embedded: int):
        super(FeedForward, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(n_embedded, 4 * n_embedded),
            nn.ReLU(),
            nn.Linear(n_embedded * 4, n_embedded),
            nn.Dropout(Config.dropout)
        )

    def forward(self, x):
        return self.m(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embedded: int, head_size: int, chunk: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.attentions = nn.ModuleList(
            [Head(n_embedded=n_embedded, head_size=head_size, chunk=chunk) for _ in range(num_heads)])

        self.fc0 = nn.Linear(
            n_embedded, n_embedded
        )
        self.dp = nn.Dropout(Config.dropout)

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor):
        if q is not None and v is not None:
            return self.dp(self.fc0(torch.cat([h(k, q, v) for h in self.attentions], dim=-1)))
        else:
            x = k
            return self.dp(self.fc0(torch.cat([h(x) for h in self.attentions], dim=-1)))


class Block(nn.Module):
    def __init__(self, n_embedded: int, n_head: int, chunk: int):
        super(Block, self).__init__()
        assert n_head // n_embedded == 0, f'n_head // n_embedded is Failed {n_head} // {n_embedded} == [{n_head // n_embedded}]'
        head_size = n_embedded // n_head
        self.block = nn.ModuleDict(
            dict(
                ln1=nn.LayerNorm(n_embedded),
                ln2=nn.LayerNorm(n_embedded),
                m1=MultiHeadAttention(n_embedded=n_embedded, chunk=chunk, head_size=head_size, num_heads=n_head),
                feed_forward=FeedForward(n_embedded=n_embedded)
            )
        )

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor):
        if q is not None and v is not None:
            x = v + self.block.ln1(self.block.m1(k, q, v))
            x = x + self.block.ln2(self.block.feed_forward(x))
            return x
        else:
            x = k
            x = x + self.block.ln1(self.block.m1(x))
            x = x + self.block.ln2(self.block.feed_forward(x))
            return x


class PTTDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, n_head: int, n_embedded: int, head_size: int, chunk: int):

        super(PTTDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_head
        self.n_embedded = n_embedded
        self.head_size = head_size
        self.chunk = chunk
        self.model = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embedded),
                wpe=nn.Embedding(vocab_size, n_embedded),
                m=nn.ModuleList([Block(n_embedded=n_embedded, n_head=n_head, chunk=chunk) for _ in range(n_layers)]),
                ln=nn.LayerNorm(n_embedded),
                fc=nn.Linear(n_embedded, vocab_size)
            )
        )
        self.model.fc.weight = self.model.wte.weight
        print(f'\033[1;32mCreated With {sum(v.numel() for v in self.parameters()) / 1e6} Million Parameters')

    def forward(self, x, target: typing.Optional[torch.Tensor] = None):

        b, t, c = x.shape
        tkm = self.model.wte(x)
        pom = self.model.wpe(torch.arange(t, dtype=torch.long))
        x = self.model.fc(self.model.ln(self.model.m(tkm + pom)))

        if target is not None:

            x = x.view(b * t, c)
            loss = nn.functional.cross_entropy(x, target.view(-1))
        else:

            loss = None
        return x, loss

    @torch.no_grad()
    def generate(self, idx, num_generate: int = 1):
        for _ in range(num_generate):
            idx_chunk = idx[:, -self.chunk:]
            logits, _ = self.forward(idx_chunk)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            next_idx = torch.multinomial(logits, 1)
            idx = torch.cat([idx, next_idx], 1)
        return idx


# class