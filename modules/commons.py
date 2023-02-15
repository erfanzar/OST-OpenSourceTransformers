from dataclasses import dataclass
from typing import Optional

from .activations import get_activation

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

__all__ = ['MultiHeadBlock', 'MultiHeadAttention', 'Head', 'FeedForward', 'Decoder', 'Encoder', 'CasualBlock',
           'PGTBlock', 'Conv1D']


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


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
        self.ln1 = LayerNorm(number_of_embedded)
        self.ln2 = LayerNorm(number_of_embedded)

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
        self.ln1 = LayerNorm(number_of_embedded)
        self.sc = CausalSelfAttention(number_of_embedded=number_of_embedded, number_of_head=number_of_head)
        self.ln2 = LayerNorm(number_of_embedded)
        self.mlp = MLP(number_of_embedded=number_of_embedded)

    def forward(self, x):
        x = x + self.sc(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


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
        # print(x.shape)
        # print(self.tensor.shape)
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
        b, t, c = k.shape
        k = self.key(k)
        q = self.queries(q)
        v = self.value(v)

        k = k.view(b, t, self.number_of_heads, self.c).transpose(1, 2)
        q = q.view(b, t, self.number_of_heads, self.c).transpose(1, 2)
        v = v.view(b, t, self.number_of_heads, self.c).transpose(1, 2)

        # DotScale
        attn = q @ k.transpose(-2, -1) * (math.sqrt(self.c))
        # print(f'ATTN : {attn.shape} ')
        # print(f'MASK : {mask.shape}')
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dp(attn)

        attn = attn @ v

        attn = self.fc(attn.transpose(1, 2).contiguous().view(b, t, c))
        return attn


class FFD(nn.Module):
    def __init__(self, embedded: int):
        super(FFD, self).__init__()
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
        self.ln1 = LayerNorm(embedded)
        self.attn = SelfAttention(embedded, number_of_heads)
        self.ln2 = LayerNorm(embedded)
        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.ff = FFD(embedded)

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
        self.ln = LayerNorm(embedded)

    def forward(self, x, src_mask):
        # print('-' * 20)
        # print(f'INPUT TO DECODER : {x.shape}')
        x = self.position(self.token(x))
        # print(f'TOKENS : {x.shape}')
        # print('-' * 20)
        for i, m in enumerate(self.layers):
            # print(f'RUNNING ENCODER {i} : {x.shape}')
            x = m(x, src_mask)
        return self.ln(x)


class DecoderLayer(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(DecoderLayer, self).__init__()
        self.ln1 = LayerNorm(embedded)
        self.ln2 = LayerNorm(embedded)
        self.ln3 = LayerNorm(embedded)

        self.attn1 = SelfAttention(embedded, number_of_heads)
        self.attn2 = SelfAttention(embedded, number_of_heads)

        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.dp3 = nn.Dropout(Conf.Dropout)
        self.ff = FFD(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        lx = self.ln1(x)
        x = self.dp1(self.attn1(lx, lx, lx, trg_mask)) + x
        lx = self.ln2(x)
        x = self.dp2(self.attn2(lx, enc_out, enc_out, src_mask)) + x
        lx = self.ln3(x)
        x = self.dp3(self.ff(lx)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int,
                 ):
        super(Decoder, self).__init__()
        self.embedded = embedded
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        self.layers = nn.ModuleList([DecoderLayer(embedded, number_of_heads) for _ in range(number_of_layers)])
        self.fc = nn.Linear(embedded, embedded)
        self.token = Embedding(vocab_size, embedded)
        self.position = PositionalEncoding(max_length, embedded)
        self.ln = LayerNorm(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.position(self.token(x))
        for m in self.layers:
            x = m(x, enc_out, src_mask, trg_mask)
        return self.fc(self.ln(x))


# =========================================================> PGT => models

@dataclass
class Config:
    num_embedding: int = 512
    num_heads: int = 8
    max_len: int = 256
    vocab_size: int = 5000
    num_layers: int = 2
    scale_attn_by_layer_idx: bool = False
    use_mask: bool = True
    attn_dropout: float = 0.2
    residual_dropout: float = 0.2
    activation = 'new_gelu'
    hidden_size: int = num_embedding
    max_position_embeddings = max_len
    embd_pdrop: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    intermediate_size: int = num_embedding * 4


class Conv1D(nn.Module):
    def __init__(self, c1, c2):
        super(Conv1D, self).__init__()
        self.c2 = c2
        w = torch.empty(c1, c2)
        nn.init.normal_(w, std=0.2)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(c2))

    def forward(self, x):
        new_shape = x.size()[:-1] + (self.c2,)
        # print(f'income : {x.shape}')
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(new_shape)
        # print(f'output : {x.shape}')
        return x


class MultiCNNAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(MultiCNNAttention, self).__init__()
        self.layer_idx = layer_idx
        self.embedding = config.hidden_size
        self.num_heads = config.num_heads
        self.num_div = self.embedding // self.num_heads
        self.scale_attn_by_layer_idx = config.scale_attn_by_layer_idx
        self.use_mask = config.use_mask
        if self.num_heads // self.embedding != 0:
            raise ValueError(
                f'hidden_size must be dividable to num_heads {self.num_heads} // {self.embedding} = {self.num_heads // self.embedding}'
            )
        self.c_attn = Conv1D(self.embedding, self.embedding * 3)
        self.c_proj = Conv1D(self.embedding, self.embedding)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.register_buffer('bias', torch.tril(
            torch.ones(config.max_len, config.max_len, dtype=torch.uint8, device=config.device).view(1, 1,
                                                                                                     config.max_len,
                                                                                                     config.max_len)))

        self.register_buffer('masked_bias', torch.tensor(float(-1e4)))

    def _split_heads(self, tensor: torch.Tensor):
        new_shape = tensor.size()[:-1] + (self.num_heads, self.num_div)
        # print(f'Shape : {new_shape}')
        tensor = tensor.view(new_shape).permute(0, 2, 1, 3)
        return tensor

    def _merge_heads(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.size()[:-2] + (self.num_heads * self.num_div,)
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value, attention_mask, head_mask):
        attn_weight = torch.matmul(query, key.transpose(-2, -1))

        attn_weight = attn_weight / torch.full([], value.size(-1) ** 0.5, dtype=attn_weight.dtype,
                                               device=attn_weight.device)
        if self.scale_attn_by_layer_idx:
            attn_weight /= self.layer_idx
        if self.use_mask:
            key_len, query_len = key.size(-2), query.size(-2)
            masked = self.bias[:, :, key_len - query_len:query_len, :key_len].to(attn_weight.device)
            attn_weight = attn_weight.masked_fill(masked == 0, self.masked_bias)
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weight = attn_weight + attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn_weight = attn_weight.type(value.dtype)
        if head_mask is not None:
            attn_weight = attn_weight * head_mask

        attn_weight = torch.matmul(attn_weight, value)
        return attn_weight

    def forward(self, hidden_state: Optional[torch.Tensor], attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_state).split(self.embedding, dim=len(hidden_state.shape) - 1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        attn_output = self._attn(query=query, key=key, value=value, attention_mask=attention_mask, head_mask=head_mask)
        attn_output = self.residual_dropout(self.c_proj(self._merge_heads(attn_output)))
        return attn_output


class PGTMLP(nn.Module):
    def __init__(self, config):
        super(PGTMLP, self).__init__()
        self.c_op = Conv1D(config.hidden_size, config.intermediate_size)
        self.c_proj = Conv1D(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.act = get_activation(config.activation)

    def forward(self, hidden_state):
        hidden_state = self.c_op(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.c_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class PGTBlock(nn.Module):
    def __init__(self, config, layer_idx_1=None, layer_idx_2=None):
        super(PGTBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ln3 = nn.LayerNorm(config.hidden_size)
        self.h_1 = MultiCNNAttention(config=config, layer_idx=layer_idx_1)
        self.h_2 = MultiCNNAttention(config=config, layer_idx=layer_idx_2)
        self.mlp = PGTMLP(config)

    def forward(self, hidden_state, attention_mask=None, heads_mask=None):
        residual = hidden_state
        hidden_state = self.ln1(hidden_state)
        hidden_state = self.h_1(hidden_state, attention_mask, heads_mask) + residual

        residual = hidden_state
        hidden_state = self.ln2(hidden_state)
        hidden_state = self.h_2(hidden_state, attention_mask, heads_mask) + residual

        residual = hidden_state
        hidden_state = self.ln3(hidden_state)
        hidden_state = self.mlp(hidden_state) + residual
        return hidden_state
