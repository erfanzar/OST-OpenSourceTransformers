import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Config:
    Dropout = 0.1
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
        self.dp1, self.dp2 = nn.Dropout(Config.Dropout).to(Config.device), nn.Dropout(Config.Dropout).to(Config.device)
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
        ce = 5
        self.m = nn.Sequential(
            nn.Linear(number_of_embedded, number_of_embedded * ce),
            NG(),
            nn.Linear(number_of_embedded * ce, number_of_embedded),
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
        attention = self.block.sa1(v, k, q, mask)
        x = self.dp(self.block.ln1(attention + q))
        # comment line below for original transformer block [start]
        #
        # x = self.block.sa2(x, x, x, mask)
        # x =( self.block.fd(self.block.ln3(x)) + x)
        x = (self.block.fd(self.block.ln3(x)) + x)
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
        self.attn = SelfAttention(number_of_embedded=number_of_embedded, number_of_heads=number_of_heads)
        self.ln = nn.LayerNorm(number_of_embedded)
        self.block = Block(
            number_of_embedded=number_of_embedded, number_of_heads=number_of_heads
        )
        self.ff = FeedForward(number_of_embedded=number_of_embedded)
        self.ln2 = nn.LayerNorm(number_of_embedded)
        self.dp = nn.Dropout(Config.Dropout)
        self.dp2 = nn.Dropout(Config.Dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attn(x, x, x, trg_mask)
        query = self.dp(self.ln(attention + x))
        out = self.block(value, key, query, src_mask)
        out = self.dp2(self.ln2(self.ff(out) + out))
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


class PTT(nn.Module):
    def __init__(self,
                 src_vocab_size, trg_vocab_size, src_pad_idx: int, trg_pad_idx: int, number_of_embedded: int,
                 chunk: int,
                 number_of_layers: int, number_of_heads: int, max_length: int):
        super(PTT, self).__init__()
        self.decoder = Decoder(
            number_of_embedded=number_of_embedded,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            max_length=max_length,
            trg_vocab_size=trg_vocab_size
        )
        self.encoder = Encoder(
            number_of_embedded=number_of_embedded,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            vocab_size=src_vocab_size,
            chunk=chunk,

        )
        self.src_pad_index = src_pad_idx
        self.trg_pad_index = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(trg.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        out_encoder = self.encoder(src, src_mask)
        # print('[ENCODER STATUS] : DONE !')
        # print(out_encoder.shape )
        out_decoder = self.decoder(trg, out_encoder, src_mask, trg_mask)
        return out_decoder


chunk = 24
if __name__ == "__main__":
    src_pad_idx = 0
    trg_pad_idx = 0

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 2, 0, 0, 0, 0, 0]]).to(Config.device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0, 0, 0]]).to(Config.device)

    transformer = PTT(src_vocab_size=12, trg_vocab_size=121, max_length=500, number_of_layers=8, number_of_heads=102,
                      number_of_embedded=510, chunk=100, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx).to(
        Config.device)
    print(sum(s.numel() for s in transformer.parameters()) / 1e6, " Million Parameters Are In PPT")
    optim = torch.optim.AdamW(transformer.parameters(), 3e-4)
    epochs = 1000
    losses = 0
    for i in range(epochs):
        predict = transformer.forward(x, trg)
        optim.zero_grad()
        b, t, c = predict.shape
        # predict = torch.nn.functional.softmax(predict, dim=-1)
        # predict = predict[:, -1, :]
        # predict = torch.multinomial(predict, num_samples=1)
        loss = torch.nn.functional.cross_entropy(predict.view(b * t, -1), target=trg.view(-1))
        loss.backward()
        optim.step()
        losses += loss
        print(f'\r [{i + 1}/{epochs}] | LOSS : {loss.item()} | AVG : {losses / (i + 1)}')
