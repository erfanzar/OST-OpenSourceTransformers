import dataclasses
import math

import jax
import jax.numpy as np
import time
from flax import linen as nn
from jax import jit, random
from tqdm.auto import tqdm
import logging
import jax_metrics
import einops
import optax

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

# defice random key
seed: int = 42
rk = jax.random.PRNGKey(seed=seed)
dense_kwargs = dict(kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.zeros)


def timer(func):
    def warper(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        tm = t2 - t1
        logger.info(f'{func.__name__}  ==> {tm} sec')
        return ret, tm

    return warper


@dataclasses.dataclass
class Config:
    n_heads: int = 8
    n_layers: int = 8
    vocab_size: int = 3200
    dtype_embedding: np.dtype = np.int16
    hidden_size: int = 768
    max_sentence_length: int = 256
    drop_prob: float = 0.1


@jit
def scaled_dot_product(query, key, value, bias=None, attention_mask=None):
    attn_logits = (np.matmul(query, np.swapaxes(key, -2, -1))) / math.sqrt(query.shape[-1])

    if bias is not None:
        attn_logits = np.where(bias[:, :, :attn_logits.shape[-1], :attn_logits.shape[-1]] == 0, -9e15, attn_logits)

    if attention_mask is not None:
        attn_logits += attention_mask[:, :, :attn_logits.shape[-1], :attn_logits.shape[-1]]
    score = nn.softmax(attn_logits, axis=-1)
    attention = np.matmul(score, value)
    return attention


class RotaryEmbedding(nn.Module):
    head_dim: int
    config: Config

    def setup(self) -> None:
        config = self.config
        self.inv_freq = 1 / (10000 * (np.arange(0, self.head_dim, 2) / self.head_dim))
        t = np.arange(config.max_sentence_length)
        freq = einops.rearrange('j,t->jt', t, self.inv_freq)
        self.max_seq_length_cached = config.max_sentence_length
        freq = np.concatenate([freq, freq], axis=-1)
        self.sin_cach = np.sin(freq[None, None, :, :])
        self.cos_cach = np.cos(freq[None, None, :, :])

    def __call__(self, value, seq_length=None):
        if seq_length > self.max_seq_length_cached:
            self.max_seq_length_cached = seq_length
            t = np.arange(self.max_seq_length_cached)
            freqs = einops.rearrange("i,j->ij", t, self.inv_freq)
            freq = np.concatenate([freqs, freqs], axis=-1)
            self.sin_cach = np.sin(freq[None, None, :, :])
            self.cos_cach = np.cos(freq[None, None, :, :])
        return (
            self.cos_cach[:, :, :seq_length, ...],
            self.sin_cach[:, :, :seq_length, ...],
        )


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    config: Config

    def setup(self):
        n_heads = self.config.n_heads
        hidden_size = self.config.hidden_size
        max_sentence_length = self.config.max_sentence_length
        head_dim = hidden_size // n_heads
        assert head_dim * n_heads == hidden_size
        self.bias = np.tril(np.ones((max_sentence_length, max_sentence_length))).reshape(1, 1, max_sentence_length,
                                                                                         max_sentence_length)
        self.head_dim = head_dim

        self.qkv_proj = nn.Dense(3 * hidden_size,
                                 **dense_kwargs
                                 )
        self.o_proj = nn.Dense(hidden_size,
                               **dense_kwargs)
        self.rotary = RotaryEmbedding(config=self.config, head_dim=head_dim)

    def __call__(self, x: np.array, attention_mask=None):
        batch_size, seq_length, embed_dim = x.shape

        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.config.n_heads, self.head_dim * 3)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = np.array_split(qkv, 3, axis=-1)
        sin, cos = self.rotary(x, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attention = scaled_dot_product(q, k, v, bias=self.bias, attention_mask=attention_mask) \
            .transpose(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)

        out = self.o_proj(attention)

        return out


class Block(nn.Module):
    config: Config

    def setup(self) -> None:
        self.attention = Attention(self.config)
        config = self.config
        self.up = nn.Dense(config.hidden_size * 6, **dense_kwargs)
        self.gate = nn.Dense(config.hidden_size * 6, **dense_kwargs)
        self.down = nn.Dense(config.hidden_size, **dense_kwargs)
        self.act = nn.silu
        self.pre_ln = nn.RMSNorm()
        self.post_ln = nn.RMSNorm()

    def __call__(self, x: np.array, train=True, attention_mask=None):
        pre_norm = self.pre_ln(x)
        attn = self.attention(pre_norm, attention_mask=attention_mask) + x
        hidden = self.post_ln(attn)
        gate = self.gate(hidden)
        act_gate = self.act(gate)
        up = self.up(hidden)
        gu = act_gate * up
        hidden = self.down(gu)
        hidden = hidden_size + attn
        return hidden


class LGeM(nn.Module):
    config: Config

    def setup(self) -> None:
        config: Config = self.config
        self.embedding = nn.Embed(config.vocab_size, config.hidden_size)
        self.blocks = [
            Block(config=config) for _ in range(config.n_layers)
        ]
        self.out = nn.Dense(config.vocab_size, **dense_kwargs)

    def __call__(self, input_ids, train=True, attention_mask=None):
        hidden = self.embedding(input_ids).reshape(1, -1, self.config.hidden_size)
        for block in self.blocks:
            hidden = block(hidden, train=train, attention_mask=attention_mask)
        return self.out(hidden)


@jax.jit
def cross_entropy(params, x, targets):
    logits = model.apply(params, x)
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
    targets = jax.nn.one_hot(targets, logits.shape[-1])
    targets = targets[..., 1:, :]
    logits = logits[..., :-1, :]
    loss = np.sum(optax.softmax_cross_entropy(logits=logits, labels=targets))

    return loss
