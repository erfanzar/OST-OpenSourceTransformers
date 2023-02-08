import typing
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import MultiHeadBlock, CasualBlock, Decoder, Encoder

__all__ = ['PTTDecoder', 'PTT', 'PTTGenerative']


class PTTDecoder(nn.Module):
    def __init__(self, vocab_size: int, number_of_layers: int, number_of_embedded: int, head_size: int,
                 number_of_head: int,
                 chunk_size: int

                 ):

        super(PTTDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.chunk = chunk_size
        self.head_size = head_size

        self.token_embedding = nn.Embedding(vocab_size, number_of_embedded)
        self.position_embedding = nn.Embedding(chunk_size, number_of_embedded)

        self.blocks = nn.Sequential(
            *[MultiHeadBlock(number_of_embedded=number_of_embedded,
                             number_of_head=number_of_head) for _
              in range(number_of_layers)])

        self.ln_f = nn.LayerNorm(number_of_embedded)  # final layer norm
        self.lm_head = nn.Linear(number_of_embedded, vocab_size)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = pos_emb + tok_emb
        x = self.blocks(x, x, x)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.chunk:]

            token, loss = self(idx_cond)

            token = token[:, -1, :]
            probs = F.softmax(token, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], 1)

        return idx


class PTTCasualHeadAttention(nn.Module):
    def __init__(self, vocab_size: int, number_of_head: int, number_of_embedded: int, number_of_layers: int,
                 chunk_size: int):
        super(PTTCasualHeadAttention, self).__init__()
        self.number_of_head = number_of_head
        self.number_of_embedded = number_of_embedded
        self.number_of_layers = number_of_layers

        self.m = nn.ModuleDict(
            dict(
                wt=nn.Embedding(vocab_size, number_of_embedded),
                wp=nn.Embedding(chunk_size, number_of_embedded),
                dropout=nn.Dropout(0.2),
                h=nn.ModuleList(
                    [CasualBlock(number_of_embedded=number_of_embedded, number_of_head=number_of_head) for _ in
                     range(number_of_layers)]),

                ln_f=nn.LayerNorm(number_of_embedded)
            )
        )
        self.ll = nn.Linear(number_of_embedded, vocab_size)
        self.m.wt.weight = self.ll.weight

    def forward(self, x, targets: typing.Optional[torch.Tensor] = None):
        device = x.device
        B, T = x.shape
        token = self.m.wt(x)
        pos = self.n.wp(torch.arange(T, dtype=torch.long if device == 'cuda' else torch.int).to(device))

        x = self.m.dropout(token + pos)
        for block in self.m.h:
            x = block(x)
        x = self.m.ln_f(x)
        if targets is not None:
            logits = self.ll(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.ll(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


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
        c = (x != self.pad_index).unsqueeze(0)
        c = c.float().masked_fill(c == 0, float('-inf')).masked_fill(c == 1, float(0.0))
        return c.to(x.device)

    def make_mask_trg(self, trg):
        trg_pad_mask = (trg != self.pad_index).unsqueeze(1)

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
    def __init__(self, vocab_size: int, chunk: int, embedded: int, number_of_heads: int, number_of_layers: int,
                 pad_index: int, eos: int):
        super(PTTGenerative, self).__init__()
        self.chunk = chunk
        self.eos = eos
        self.enc = Encoder(vocab_size, chunk, embedded, number_of_heads, number_of_layers)
        self.dec = Decoder(vocab_size, chunk, embedded, number_of_heads, number_of_layers)
        self.fc = nn.Linear(embedded, vocab_size)
        self.pad_index = pad_index

    def forward_encoder(self, x, src_mask):
        return self.dec(x, src_mask)

    def make_mask_trg(self, trg):
        # print(src.shape)
        # trg_pad_mask = (trg != self.pad_index).unsqueeze(1).unsqueeze(2)
        # trg_len = trg.shape[1]
        # trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        # trg_mask = trg_pad_mask & trg_sub_mask

        trg_pad_mask = (trg != self.pad_index).unsqueeze(1).unsqueeze(2).bool()
        sq_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((sq_len, sq_len), device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask.to(trg.device)

    def make_mask_src(self, x):

        c = (x != self.pad_index).unsqueeze(1).unsqueeze(2)
        c = c.float().masked_fill(c == 0, float('-inf')).repeat(1, 1, x.shape[1], 1)

        return c.to(x.device)

    def forward(self, src, trg, target=None):
        global b
        if len(src.shape) == 3:
            b, t, c = src.shape
        else:
            b = src.shape[0]
        src_mask = self.make_mask_src(src)
        trg_mask = self.make_mask_trg(trg)
        enc = self.enc(src, src_mask)

        pred = self.dec(trg, enc, src_mask, trg_mask)

        if target is not None:
            pred = self.fc(pred)
            # print(pred.shape)
            target = target.reshape(b, -1)
            pred_l = pred.view(b, -1, pred.size(-1))
            # pred_l = F.softmax(pred_l.permute(0, 2, 1), dim=-1)
            loss = 0
            for i in range(b):
                loss += F.cross_entropy(pred_l[i], target[i], ignore_index=self.pad_index)
        else:
            pred = self.fc(pred[:, [-1], :])
            loss = None
        return pred, loss

    @torch.no_grad()
    def generate(self, src, idx, trg=None, temp=1.0):
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)

        for i in range(idx.shape[-1] - 1):
            idx = idx[:, -self.chunk:]
            pred, _ = self.forward(src, idx, target=trg)
            pred = pred[:, -1, :] / temp
            # print('TRG CHUNK : ', idx)
            pred = F.softmax(pred, dim=-1)
            next_index = torch.multinomial(pred, 1)
            # idx = torch.cat([idx, torch.zeros(1, 1, device=idx.device, dtype=torch.long)], 1)
            index = (i + 1) % self.chunk
            # print(index)
            idx[:, index] = next_index
            # print('TRG WORD : ', idx)
            if next_index == self.eos:
                break
        return idx
