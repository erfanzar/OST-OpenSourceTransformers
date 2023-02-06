import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Conf:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Dropout = 0.2


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedded: int):
        super(Embedding, self).__init__()
        self.m = nn.Embedding(vocab_size, embedded)

    def forward(self, x):
        return self.m(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, embedded: int):
        super(PositionalEncoding, self).__init__()
        tensor = torch.zeros((max_length, embedded))
        self.embedded = embedded
        for pos in range(max_length):
            for i in range(0, embedded, 2):
                tensor[pos, i] = math.sin(pos / (10_000 ** ((2 * i) / embedded)))
                tensor[pos, i + 1] = math.cos(pos / (10_000 ** ((2 * (i + 1)) / embedded)))
        self.register_buffer('tensor', tensor)

    def forward(self, x):
        x = x * math.sqrt(self.embedded)
        print(x.shape)
        print(self.tensor.shape)
        max_length = x.size(1)
        x = x + torch.autograd.Variable(self.tensor[:max_length, :], requires_grad=False)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(SelfAttention, self).__init__()
        c = embedded // number_of_heads
        assert (c * number_of_heads == embedded)
        self.c = c
        self.embedded = embedded
        self.number_of_heads = number_of_heads
        self.key = nn.Linear(embedded, embedded, bias=False)
        self.queries = nn.Linear(embedded, embedded, bias=False)
        self.value = nn.Linear(embedded, embedded, bias=False)
        self.fc = nn.Linear(embedded, embedded)
        self.dp = nn.Dropout()

    def forward(self, k, q, v, mask=None):
        shape_s = k.shape
        b = k.shape[0]
        k = self.key(k)
        q = self.queries(q)
        v = self.value(v)

        k = k.view(b, -1, self.number_of_heads, self.c).permute(0, 2, 1, 3)
        q = q.view(b, -1, self.number_of_heads, self.c).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.number_of_heads, self.c).permute(0, 2, 1, 3)

        # DotScale
        attn = q @ k.transpose(-2, -1) * (math.sqrt(self.c))
        # print(f'MASK : {mask.shape}')
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dp(attn)
        # print(f'ATTN : {attn.shape}')
        # print(f'VALUE : {v.shape}')
        attn = attn @ v
        # print(f'SCORE : {attn.shape}')
        attn = attn.transpose(1, 2).contiguous().view(shape_s)
        return self.fc(attn)


class FeedForward(nn.Module):
    def __init__(self, embedded: int):
        super(FeedForward, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(embedded, embedded * 4),
            nn.ReLU(),
            nn.Dropout(Conf.Dropout),
            nn.Linear(4 * embedded, embedded)
        )

    def forward(self, x):
        return self.m(x)


class EncoderLayer(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(EncoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(embedded)
        self.attn = SelfAttention(embedded, number_of_heads)
        self.ln2 = nn.LayerNorm(embedded)
        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.ff = FeedForward(embedded)

    def forward(self, x, src_mask):
        xl = self.ln1(x)
        ka = self.dp1(self.attn(xl, xl, xl, src_mask))
        # print(f'KA DIM : {ka.shape}')
        x = ka + x
        xl = self.ln2(x)
        x = self.dp2(self.ff(xl)) + x
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int):
        super(Encoder, self).__init__()
        self.embedded = embedded
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        self.layers = nn.ModuleList([EncoderLayer(embedded, number_of_heads) for _ in range(number_of_layers)])

        self.token = Embedding(vocab_size, embedded)
        self.position = PositionalEncoding(max_length, embedded)
        self.ln = nn.LayerNorm(embedded)

    def forward(self, x, src_mask):
        print('-' * 20)
        print(f'INPUT TO DECODER : {x.shape}')
        x = self.position(self.token(x))
        print(f'TOKENS : {x.shape}')
        print('-' * 20)
        for i, m in enumerate(self.layers):
            # print(f'RUNNING ENCODER {i} : {x.shape}')
            x = m(x, src_mask)
        return self.ln(x)


class DecoderLayer(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(DecoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(embedded)
        self.ln2 = nn.LayerNorm(embedded)
        self.ln3 = nn.LayerNorm(embedded)

        self.attn1 = SelfAttention(embedded, number_of_heads)
        self.attn2 = SelfAttention(embedded, number_of_heads)

        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.dp3 = nn.Dropout(Conf.Dropout)
        self.ff = FeedForward(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        lx = self.ln1(x)
        x = self.dp1(self.attn1(lx, lx, lx, trg_mask)) + x
        lx = self.ln2(x)
        x = self.dp2(self.attn2(lx, enc_out, enc_out, src_mask)) + x
        lx = self.ln3(x)
        x = self.dp3(self.ff(lx)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int):
        super(Decoder, self).__init__()
        self.embedded = embedded
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        self.layers = nn.ModuleList([DecoderLayer(embedded, number_of_heads) for _ in range(number_of_layers)])

        self.token = Embedding(vocab_size, embedded)
        self.position = PositionalEncoding(max_length, embedded)
        self.ln = nn.LayerNorm(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        print('-' * 20)
        print(f'INPUT TO ENCODER : {x.shape}')
        x = self.position(self.token(x))
        print(f'TOKENS : {x.shape}')
        for m in self.layers:
            x = m(x, enc_out, src_mask, trg_mask)
        return self.ln(x)


class PTT(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int,
                 pad_index: int):
        super(PTT, self).__init__()
        self.enc = Encoder(vocab_size, max_length, embedded, number_of_heads, number_of_layers)
        self.dec = Decoder(vocab_size, max_length, embedded, number_of_heads, number_of_layers)
        self.fc = nn.Linear(embedded, vocab_size)
        self.pad_index = pad_index

    def forward_encoder(self, x, enc_out, src_mask, trg_mask):
        return self.dec(x, enc_out, src_mask, trg_mask)

    def forward_decoder(self, x, src_mask):
        return self.enc(x, src_mask)

    def make_mask_src(self, x):
        c = (x != self.pad_index).unsqueeze(1).unsqueeze(2)
        return c.to(x.device)

    def make_mask_trg(self, trg):
        trg_pad_mask = (trg != self.pad_index).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask.to(trg.device)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        if trg_mask is None:
            trg_mask = self.make_mask_trg(trg)
        if src_mask is None:
            src_mask = self.make_mask_src(src)
        # x, src_mask
        # print(f'SRC : {src.shape}')
        enc = self.enc(src, src_mask)
        # x, enc_out, src_mask, trg_mask
        dec = self.dec(trg, enc, src_mask, trg_mask)
        pred = self.fc(dec)
        return pred


class PTTGenerative(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int,
                 pad_index: int):
        super(PTTGenerative, self).__init__()
        self.enc = Encoder(vocab_size, max_length, embedded, number_of_heads, number_of_layers)
        # self.dec = Decoder(vocab_size, max_length, embedded, number_of_heads, number_of_layers)
        self.fc = nn.Linear(embedded, vocab_size)
        self.pad_index = pad_index

    def forward_encoder(self, x, src_mask):
        return self.dec(x, src_mask)

    def make_mask(self, src):
        src_pad_mask = (src != self.pad_index).unsqueeze(1).unsqueeze(2)

        src_len = src.shape[1]

        src_sub_mask = torch.tril(torch.ones((src_len, src_len), device=src.device)).bool()

        src_mask = src_pad_mask & src_sub_mask

        return src_mask.to(src.device)

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.make_mask_src(src)
        enc = self.enc(src, src_mask)
        pred = self.fc(enc)
        return pred
