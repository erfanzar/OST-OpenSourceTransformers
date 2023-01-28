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
    def __init__(self, c: int = 32, head_size: int = 16, chunk: int = 8):
        super(Head, self).__init__()
        self.head_size = head_size
        self.key = nn.Linear(c, head_size, bias=False)
        self.query = nn.Linear(c, head_size, bias=False)
        self.value = nn.Linear(c, head_size, bias=False)
        self.register_buffer('base', torch.tril(torch.ones(chunk, chunk)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.base[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


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
    def __init__(self, num, head_size: int, chunk_size: int, number_of_embedded: int):
        super(MultiHeadAttention, self).__init__()

        self.m = nn.ModuleList([Head(head_size=head_size, chunk=chunk_size, c=number_of_embedded) for _ in range(num)])
        self.proj = nn.Linear(number_of_embedded, number_of_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.m], dim=-1)
        x = self.dp(self.proj(x))
        return x


class MultiHeadBlock(nn.Module):
    def __init__(self, number_of_head, number_of_embedded: int, chunk_size: int = 8):
        super(MultiHeadBlock, self).__init__()
        head_size = number_of_embedded // number_of_head
        self.sa = MultiHeadAttention(number_of_head, head_size=head_size, chunk_size=chunk_size,
                                     number_of_embedded=number_of_embedded)
        self.ffwd = FeedForward(number_of_embedded=number_of_embedded)
        self.ln1 = nn.LayerNorm(number_of_embedded)
        self.ln2 = nn.LayerNorm(number_of_embedded)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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
