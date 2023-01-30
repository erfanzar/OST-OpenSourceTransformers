import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Config:
    Dropout = 0.2
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'


class SelfAttention(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_heads):
        super(SelfAttention, self).__init__()
        head_size = number_of_embedded // number_of_heads
        # assert (head_size * number_of_heads == number_of_embedded)
        self.key = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.query = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.value = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.fc_out = nn.Linear(number_of_embedded, number_of_embedded).to(Config.device)
        self.dp1, self.dp2 = nn.Dropout(Config.Dropout).to(Config.device), nn.Dropout(Config.Dropout).to(Config.device)
        self.head_size = head_size
        self.number_of_embedded = number_of_embedded
        self.number_of_heads = number_of_heads

    def forward(self, v, k, q, mask: Optional[torch.Tensor] = None):
        b, t, c = k.shape

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)
        k = k.reshape(b, t, self.number_of_heads, self.head_size).transpose(1, 2)
        # (batch, chunk , number_of_heads, head_size) -> (batch, number_of_heads, chunk, head_size)
        q = q.reshape(b, t, self.number_of_heads, self.head_size).transpose(1, 2)
        # (batch, chunk , number_of_heads, head_size) -> (batch, number_of_heads, chunk, head_size)
        v = v.reshape(b, t, self.number_of_heads, self.head_size).transpose(1, 2)
        # (batch, chunk , number_of_heads, head_size) -> (batch, number_of_heads, chunk, head_size)

        score = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(0)))
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = torch.nn.functional.softmax(score, dim=-1)
        score = self.dp1(score)
        out = score @ v

        # score = score.transpose(1, 2).contiguous().view(b, t, c)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.dp2(self.fc_out(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, number_of_embedded: int):
        super(FeedForward, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(number_of_embedded, number_of_embedded * 4),
            nn.ReLU(),
            nn.Linear(number_of_embedded * 4, number_of_embedded),
            nn.Dropout(Config.Dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.m(x)


class Block(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_heads: int):
        super(Block, self).__init__()
        self.block = nn.ModuleDict(
            dict(
                sa1=SelfAttention(number_of_embedded, number_of_heads),
                ln1=nn.LayerNorm(number_of_embedded),
                ln2=nn.LayerNorm(number_of_embedded),
                sa2=SelfAttention(number_of_embedded, number_of_heads),
                fd=FeedForward(number_of_embedded),
                ln3=nn.LayerNorm(number_of_embedded),
            )
        )
        self.dp1 = nn.Dropout(Config.Dropout)
        self.dp2 = nn.Dropout(Config.Dropout)
        self.dp3 = nn.Dropout(Config.Dropout)

    def forward(self, v, k, q, mask):
        attention = self.block.sa1(v, k, q, mask)
        x = self.dp1(self.block.ln1(attention + q))
        # comment line below for original transformer block [start]

        x = self.block.sa2(x, x, x, None)
        x = self.dp2(self.block.ln2(x) + x)

        # [end]
        x = self.dp3(self.block.ln3(self.block.fd(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, chunk: int, number_of_embedded: int, number_of_layers: int,
                 number_of_heads: int):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, number_of_embedded)
        self.position_embedding = nn.Embedding(chunk, number_of_embedded)
        self.blocks = nn.ModuleList(
            [Block(number_of_embedded, number_of_heads) for _ in range(number_of_layers)]
        )
        self.dp = nn.Dropout(Config.Dropout)
        self.fc = nn.Linear(number_of_embedded, vocab_size)
        self.fc.weight = self.token_embedding.weight
        self.dp = nn.Dropout(Config.Dropout)

    def forward(self, x, mask=None):
        b, t = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(
            torch.arange(t, dtype=torch.long if x.device == 'cuda' else torch.int).to(x.device)).view(b, t, -1)
        print(pos_emb.shape)
        print(token_emb.shape)
        att = self.dp(pos_emb + token_emb)
        print(f'ATT : {att.shape}')
        for i, m in enumerate(self.blocks):
            att = m(att, att, att, mask)
            print(f'[{i}] = {att.shape}')
        return att


chunk = 24
if __name__ == "__main__":
    block = Encoder(vocab_size=1002, number_of_embedded=2040, number_of_layers=3, chunk=chunk * 5,
                    number_of_heads=2, ).to(
        Config.device)
    print('Compiling >>> .. ')
    block = torch.compile(block)
    print('Compiled * ')

    inputs = torch.ones(1, chunk, dtype=torch.long).to(Config.device)
    sk = torch.tril(torch.ones(1, 1, chunk, chunk))[:, :, :chunk, :chunk].to(Config.device)
    v = block(inputs, sk)
    print(f'VShape : {v.shape}')
    # print(f'ScoreShape : {cs.shape}')
    v = torch.softmax(v, dim=-1)
    v = v[:, -1, :]
    print(v.shape)
    v = torch.multinomial(v, num_samples=1)
    print(v)
    print(sum(s.numel() for s in block.parameters()) / 1e6)
