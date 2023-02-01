import math
import torch
import torch.nn as nn
from torch.nn import functional as f


class SelfAttention(nn.Module):
    def __init__(self, ne: int, nh: int):
        super(SelfAttention, self).__init__()
        self.nh = nh
        self.ne = ne
        self.hs = ne // nh
        assert (self.hs * self.nh) == self.ne
        self.key = nn.Linear(ne, ne, bias=False)
        self.query = nn.Linear(ne, ne, bias=False)
        self.value = nn.Linear(ne, ne, bias=False)
        self.fc = nn.Linear(ne, ne, bias=True)

    def forward(self, query, key, value, mask=None):
        b, t, c = query.shape
        query = self.query(query).view(b, -1, self.nh, self.hs).transpose(1, 2)
        key = self.key(key).view(b, -1, self.nh, self.hs).transpose(1, 2)
        value = self.value(value).view(b, -1, self.nh, self.hs).transpose(1, 2)

        attn = query @ key.transpose(-1, -2) * (math.sqrt(self.hs))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = f.softmax(attn, dim=-1)
        value = attn @ value
        value = value.transpose(1, 2).contiguous().view(b, -1, self.ne)
        return self.fc(value)


class PositionalEncoding(nn.Module):
    def __init__(self, ne: int, max_len: int = 120):
        super(PositionalEncoding, self).__init__()
        self.ne = ne
        pos = torch.zeros(max_len, ne)
        for p in range(max_len):
            for i in range(0, ne, 2):
                pos[p, i] = math.sin(pos / (10000 ** ((2 * i) / ne)))
                pos[p, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / ne)))
        pos = pos.unsqueeze(0)
        self.register_buffer('p', pos)

    def forward(self, x):
        x = x + torch.sqrt(self.ne)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.p[:, :seq_len], requires_grad=False).cuda()
        return x


class FeedForward(nn.Module):
    def __init__(self, ne: int, dropout: float = 0.2):
        super(FeedForward, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(ne, ne * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ne * 4, ne)
        )

    def forward(self, x):
        return self.m(x)


class Norm(nn.Module):
    def __init__(self, ne: int, eps=1e-6):
        super().__init__()

        self.size = ne
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderBlock(nn.Module):
    def __init__(self, ne: int, nh: int, dropout: float = 0.2):
        super(EncoderBlock, self).__init__()
        self.ne = ne
        self.nh = nh
        self.ln1 = Norm(ne)
        self.ln2 = Norm(ne)

        self.attn = SelfAttention(ne=ne, nh=nh)
        self.fd = FeedForward(ne, dropout)

        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        n = self.ln1(x)
        x = self.do1(self.attn(n, n, n, src_mask)) + x
        n = self.ln2(x)
        x = self.do2(self.fd(n)) + x
        return x


class DecoderBlock(nn.Module):
    def __init__(self, ne: int, nh: int, dropout: float = 0.2):
        super(DecoderBlock, self).__init__()
        self.ln1 = Norm(ne)
        self.ln2 = Norm(ne)
        self.ln3 = Norm(ne)

        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)

        self.attn1 = SelfAttention(ne=ne, nh=nh)
        self.attn2 = SelfAttention(ne=ne, nh=nh)

        self.fd = FeedForward(ne, dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        n = self.ln1(x)
        x = self.do1(self.attn1(n, n, n, trg_mask)) + x
        n = self.ln2(x)
        x = self.do2(self.attn2(n, enc_out, enc_out, src_mask)) + x
        n = self.ln3(x)
        x = self.do3(self.fd(n)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, ne: int, nh: int, nl: int, max_length: int):
        super(Decoder, self).__init__()

        self.m = nn.ModuleList(
            [
                DecoderBlock(ne=ne, nh=nh) for _ in nl
            ]
        )
        self.te = nn.Embedding(vocab_size, ne)
        self.pe = PositionalEncoding(ne=ne, max_len=max_length)
        self.norm = Norm(ne)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        x = self.te(x)
        x = self.pe(x)
        for m in self.m:
            x = m(x, enc_out, trg_mask, src_mask)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, ne: int, nh: int, nl: int, max_length: int):
        super(Encoder, self).__init__()
        self.norm = Norm(ne)
        self.m = nn.ModuleList(
            [
                EncoderBlock(ne=ne, nh=nh) for _ in nl
            ]
        )
        self.te = nn.Embedding(vocab_size, ne)
        self.pe = PositionalEncoding(ne=ne, max_len=max_length)

    def forward(self, x, src_mask=None):
        x = self.pe(self.te(x))
        for m in self.m:
            x = m(x, src_mask)
        return self.norm(x)


class PTT(nn.Module):
    def __init__(self, vocab_size: int, ne: int, nh: int, nl: int, max_length: int):
        super(PTT, self).__init__()
        self.enc = Encoder(vocab_size=vocab_size, ne=ne, nh=nh, nl=nl, max_length=max_length)
        self.dec = Decoder(vocab_size=vocab_size, ne=ne, nh=nh, nl=nl, max_length=max_length)
        self.fc = nn.Linear(ne, vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        enc = self.enc(src, src_mask)
        return self.fc(self.dec(trg, enc, src_mask, trg_mask))
