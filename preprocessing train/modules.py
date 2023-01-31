import math
from dataclasses import dataclass
from typing import Optional

import erutils
import sentencepiece
import torch
import torch.nn as nn


def detokenize_words(word: list, first_word_token: int = 0, last_word_token: int = 1002, pad_index: int = 1003):
    """
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: un tokenized words
    """

    w = [(first_word_token if w == last_word_token - 1 else w) for w in
         [w for w in word if w not in [last_word_token, first_word_token]]]
    del w[-1]
    # print(f'W : {w}')
    return w


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
    def __init__(self, vocab_size: int, max_length: int, number_of_embedded: int, number_of_layers: int,
                 number_of_heads: int):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, number_of_embedded)
        self.position_embedding = nn.Embedding(max_length, number_of_embedded)
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
            max_length=max_length,

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

    def forward_encoder(self, src):
        src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask), src_mask

    def forward_decoder(self, trg, encoder_output, src_mask):
        trg_mask = self.make_trg_mask(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def forward(self, src, trg):
        enc, src_mask = self.forward_encoder(src)
        out = self.forward_decoder(trg, enc, src_mask)
        return out


def tokenize_words(word: list, first_word_token: int = 0, swap: int = 1001, last_word_token: int = 1002,
                   pad_index: int = 1003):
    """
    :param swap:
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: 0 for start token | 1002 for end token
    """
    word = [(swap if w == 0 else w) for w in word]
    word = [first_word_token] + word
    word.append(last_word_token)
    word.append(pad_index)
    return word


sentence = sentencepiece.SentencePieceProcessor()
sentence.Load(model_file='../tokenizer_model/test_model.model')


def fix_data(data):
    # data = itertools.islice(data.items(), 500)
    for d in data:
        question = data[d]['question']
        answers = data[d]['answers']
        encoded_question = tokenize_words(sentence.Encode(question))
        encoded_answers = tokenize_words(sentence.Encode(answers))
        yield encoded_question, encoded_answers


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)


if __name__ == "__main__":

    data = erutils.read_json('../data/train-v2.0-cleared.json')

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 2, 0, 0, 0, 0, 0]]).to(Config.device)
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0, 0, 0]]).to(Config.device)

    transformer = PTT(src_vocab_size=sentence.vocab_size() + 4, trg_vocab_size=sentence.vocab_size() + 4,
                      max_length=1000,
                      number_of_layers=8,
                      number_of_heads=12,
                      number_of_embedded=756, chunk=100, src_pad_idx=1003, trg_pad_idx=1003).to(
        Config.device)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print(sum(s.numel() for s in transformer.parameters()) / 1e6, " Million Parameters Are In MODEL")
    optim = torch.optim.AdamW(transformer.parameters(), 4e-4, betas=(0.9, 0.98), eps=1e-9)
    epochs = 1000
    losses_t = 0
    ipa = 0
    for i in range(epochs):
        losses = 0
        for ia, xt in enumerate(fix_data(data)):
            x = torch.tensor(xt[0]).to(Config.device).unsqueeze(0)

            trg = torch.tensor(xt[1]).to(Config.device).unsqueeze(0)
            predict = transformer.forward(x, trg[:, :-1])
            optim.zero_grad()
            b, t, c = predict.shape
            # predict = torch.nn.functional.softmax(predict, dim=-1)
            # predict = predict[:, -1, :]
            # predict = torch.multinomial(predict, num_samples=1)
            loss = torch.nn.functional.cross_entropy(predict.view(-1, predict.size(-1)), target=trg.view(-1)[:-1],
                                                     ignore_index=1003)
            loss.backward()
            optim.step()
            ipa += 1
            losses += loss
            losses_t += loss
            print(
                f'\r\033[1;36m [{i + 1}/{epochs}] | ITER : [{ia + 1}] | LOSS : {loss.item()} | AVG ITER : {losses / (ia + 1)} | AVG EP : {losses_t / (ipa)}',
                end='')

        print()
        if i % 10 == 0:
            save_model(model=transformer.state_dict(), optimizer=optim.state_dict(), epochs=epochs, epoch=i)
