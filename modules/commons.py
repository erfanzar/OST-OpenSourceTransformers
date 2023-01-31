from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except:
    print('Downloading Missing Module [pytorch]')
    import subprocess
    import sys

    path = sys.executable
    subprocess.run(f'{path} -m pip install torch')
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

# torch.manual_seed(1377)
import math

__all__ = ['MultiHeadBlock', 'MultiHeadAttention', 'Head', 'FeedForward']


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Head(nn.Module):
    def __init__(self, n_embedded: int, head_size: int):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embedded, head_size, bias=False)
        self.query = nn.Linear(n_embedded, head_size, bias=False)
        self.value = nn.Linear(n_embedded, head_size, bias=False)

    def forward(self, k: torch.Tensor, q: torch.Tensor = None, v: torch.Tensor = None, mask=None):
        # if q is not None and v is not None:
        assert k.shape == q.shape and q.shape == v.shape
        b, t, c = k.shape
        key = self.key(k)
        query = self.query(q)
        value = self.value(v)
        attn = query @ key.transpose(-2, -1) * c ** -0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=-1)
        value = attn @ value
        return value, attn


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, number_of_embedded: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embedded, 4 * number_of_embedded),
            nn.ReLU(),
            nn.Linear(4 * number_of_embedded, number_of_embedded),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num, head_size: int, number_of_embedded: int):
        super(MultiHeadAttention, self).__init__()

        self.m = nn.ModuleList([Head(head_size=head_size, n_embedded=number_of_embedded) for _ in range(num)])
        self.proj = nn.Linear(number_of_embedded, number_of_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.cat([h(x, x, x, torch.tril(torch.ones(x.shape[1], x.shape[1])) if i == 0 else None) for i, h in
                       enumerate(self.m)],
                      dim=-1)
        x = self.dp(self.proj(x))
        return x


class MultiHeadBlock(nn.Module):
    def __init__(self, number_of_head, number_of_embedded: int):
        super(MultiHeadBlock, self).__init__()
        head_size = number_of_embedded // number_of_head
        self.sa = MultiHeadAttention(number_of_head, head_size=head_size,
                                     number_of_embedded=number_of_embedded)
        self.ffwd = FeedForward(number_of_embedded=number_of_embedded)
        self.ln1 = nn.LayerNorm(number_of_embedded)
        self.ln2 = nn.LayerNorm(number_of_embedded)

    def forward(self, x):
        x = x + self.ln1(self.sa(x, x, x))
        x = x + self.ln2(self.ffwd(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_head: int, chunk: int):
        super(CausalSelfAttention, self).__init__()
        assert \
            number_of_embedded % number_of_head == 0, \
            'number_of_embedded % number_of_head == 0 Failed Make' \
            ' Sure that number_of_embedded is equal to number_of_head'
        self.number_of_embedded = number_of_embedded
        self.number_of_head = number_of_head
        self.attn = nn.Linear(number_of_embedded, 3 * number_of_embedded)
        self.proj = nn.Linear(number_of_embedded, number_of_embedded)
        self.register_buffer('bias', torch.tril(torch.ones(chunk, chunk).view(1, 1, chunk, chunk)))
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.number_of_embedded, dim=2)
        q = q.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)
        k = k.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)
        v = v.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) * (1.0 / torch.sqrt(k.size(0)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dp1(attn)
        attn = attn @ v
        attn = attn.transpose(2, 1).contiguous().view(B, T, C)
        attn = self.dp2(self.proj(attn))
        return attn


class MLP(nn.Module):
    def __init__(self, number_of_embedded: int):
        super(MLP, self).__init__()
        self.li1 = nn.Linear(number_of_embedded, 4 * number_of_embedded)
        self.li2 = nn.Linear(4 * number_of_embedded, number_of_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dp(self.li2(new_gelu(self.li1(x))))
        return x


class CasualBlock(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_head: int):
        super(CasualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(number_of_embedded)
        self.sc = CausalSelfAttention(number_of_embedded=number_of_embedded, number_of_head=number_of_head)
        self.ln2 = nn.LayerNorm(number_of_embedded)
        self.mlp = MLP(number_of_embedded=number_of_embedded)

    def forward(self, x):
        x = x + self.sc(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class Config:
    Dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class NG(nn.Module):
    def __init__(self):
        super(NG, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SelfAttention(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_heads):
        super(SelfAttention, self).__init__()
        head_size = number_of_embedded // number_of_heads
        assert (head_size * number_of_heads == number_of_embedded)
        self.key = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.query = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.value = nn.Linear(number_of_embedded, number_of_embedded, bias=False).to(Config.device)
        self.fc_out = nn.Linear(number_of_embedded, number_of_embedded).to(Config.device)
        self.dp1 = nn.Dropout(Config.Dropout).to(Config.device)
        self.dp2 = nn.Dropout(Config.Dropout).to(Config.device)
        self.head_size = head_size
        self.number_of_embedded = number_of_embedded
        self.number_of_heads = number_of_heads

    def forward(self, v, k, q, mask: Optional[torch.Tensor] = None):
        b = q.shape[0]
        b, t, c = q.shape
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        k = self.key(k)
        q = self.query(q)
        v = self.value(v)
        # print('-' * 40)
        # print(f'KEY : {k.shape}')
        # print(f'VALUE : {v.shape}')
        # print(f'QUERY : {q.shape}')
        k = k.reshape(b, key_len, self.number_of_heads, self.head_size).transpose(1, 2)
        # (batch, chunk , number_of_heads, head_size) -> (batch, number_of_heads, chunk, head_size)
        q = q.reshape(b, query_len, self.number_of_heads, self.head_size).transpose(1, 2)
        # (batch, chunk , number_of_heads, head_size) -> (batch, number_of_heads, chunk, head_size)
        v = v.reshape(b, value_len, self.number_of_heads, self.head_size).transpose(1, 2)
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
        ce = 4
        self.m = nn.Sequential(
            nn.Linear(number_of_embedded, number_of_embedded * ce),
            NG(),
            nn.Dropout(Config.Dropout),
            nn.Linear(number_of_embedded * ce, number_of_embedded),

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
                # Comment [Start]
                # ln2=nn.LayerNorm(number_of_embedded),
                # sa2=SelfAttention(number_of_embedded, number_of_heads),

                fd=FeedForward(number_of_embedded),
                ln3=nn.LayerNorm(number_of_embedded),
                # Comment [END]
            )
        )
        self.dp = nn.Dropout(Config.Dropout)
        # self.dp2 = nn.Dropout(Config.Dropout)

    def forward(self, v, k, q, mask):
        attention = self.block.sa1(self.block.ln1(v), self.block.ln1(k), self.block.ln1(q), mask)
        x = self.dp(attention) + q
        # comment line below for original transformer block [start]
        #
        # x = self.block.sa2(x, x, x, mask)
        # x =( self.block.fd(self.block.ln3(x)) + x)
        x = (self.block.fd(self.block.ln3(x))) + x
        # x = self.dp2(self.block.ln3(self.block.fd(x)))
        # [end]

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
        # print(x)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(
            torch.arange(t * b, dtype=torch.long if x.device == 'cuda' else torch.int).to(x.device)).view(b, t, -1)

        # print(pos_emb.shape)
        # print(token_emb.shape)
        att = self.dp(pos_emb + token_emb)
        # print(f'ATT : {att.shape}')
        for i, m in enumerate(self.blocks):
            att = m(att, att, att, mask)
            # print(f'[{i}] = {att.shape}')
        return att


class DecoderBlocK(nn.Module):
    def __init__(self, number_of_embedded, number_of_heads):
        super(DecoderBlocK, self).__init__()
        self.attn1 = SelfAttention(number_of_embedded=number_of_embedded, number_of_heads=number_of_heads)
        self.attn2 = SelfAttention(number_of_embedded=number_of_embedded, number_of_heads=number_of_heads)

        self.ln1 = nn.LayerNorm(number_of_embedded)
        self.ln2 = nn.LayerNorm(number_of_embedded)
        self.ln3 = nn.LayerNorm(number_of_embedded)

        self.ff = FeedForward(number_of_embedded=number_of_embedded)

        self.dp1 = nn.Dropout(Config.Dropout)
        self.dp2 = nn.Dropout(Config.Dropout)
        self.dp3 = nn.Dropout(Config.Dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        x = self.ln1(x)
        query = self.dp1(self.attn1(x, x, x, trg_mask)) + x
        out = self.dp2(self.attn2(self.ln2(value), self.ln2(key), self.ln2(query), src_mask)) + query
        out = self.dp2(self.ff(self.ln2(out))) + out
        return out


class Decoder(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_heads: int, number_of_layers: int, max_length: int,
                 trg_vocab_size: int):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(trg_vocab_size, number_of_embedded)
        self.position_embedding = nn.Embedding(max_length, number_of_embedded)
        self.layers = nn.ModuleList(
            [
                DecoderBlocK(number_of_embedded=number_of_embedded, number_of_heads=number_of_heads) for _ in
                range(number_of_layers)
            ]
        )
        self.fc_out = nn.Linear(number_of_embedded, trg_vocab_size)
        self.dp = nn.Dropout(Config.Dropout)

    def forward(self, x, encoder_outs, src_mask, trg_mask):
        b, t = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(
            torch.arange(t * b, dtype=torch.long if x.device == 'cuda' else torch.int).to(x.device)).view(b, t, -1)
        pps = self.dp(token_emb + pos_emb)
        for layer in self.layers:
            pps = layer(pps, encoder_outs, encoder_outs, src_mask, trg_mask)
        out = self.fc_out(pps)
        return out
