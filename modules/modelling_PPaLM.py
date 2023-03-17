from typing import List, Tuple, Any, Union, Optional

import numpy as onp
from jax import random, nn, lax, jit, numpy as np
from jax.numpy import einsum
import jax.numpy as jnp
from equinox import Module, static_field
from jax import nn as nn
from einops import rearrange, repeat
import jax


class PPaLMConfig:
    seed: Optional[int] = 42
    vocab_size: Optional[int] = -1
    hidden_size: Optional[int] = 1024
    dim_head: Optional[int] = 64
    n_layers: Optional[int] = 10
    n_heads: Optional[int] = 8
    key: Union[Any] = jax.random.PRNGKey(seed)
    up_inner_dim: Optional[int] = 4
    eps: Optional[float] = 1e-5
    mask_value: Union[int, float, Any] = 1e9


class LayerNorm(Module):
    def __init__(self, config: Optional[PPaLMConfig]):
        self.weight: Optional[jnp.array] = np.ones((config.hidden_size,))
        self.eps: Optional[float] = config.eps

    def __call__(self, x: jnp.array):
        x = x * ((1 / (jnp.sqrt(jnp.mean(x ** 2, keepdims=True)))) + self.eps)
        return x * self.weight


@jit
def fixed_pos_embedding(inv_freq, seq):
    sinusoid_inp = einsum('i , j -> i j', np.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, '... d -> ... (d r)', r=2)
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


@jit
def rotate_every_two(x):
    x = rearrange(x, '... (x v) -> ... x v', v=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = np.stack((-x2, x1), axis=-1)
    return rearrange(x, '... x v -> ... (x v)')


@jit
def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x) * sin)


class PPaLMBlock(Module):

    def __init__(self, config: Optional[PPaLMConfig]):
        hidden_size = config.hidden_size
        dim_head = config.n_layers
        n_heads = config.n_heads
        key = config.key
        up_inner_dim = config.up_inner_dim
        mask_value = config.mask_value
        attn_inner_dim = dim_head * n_heads
        ff_inner_dim = hidden_size * up_inner_dim
        self.norm: Optional[LayerNorm] = LayerNorm(hidden_size)
        self.post_norm: Optional[LayerNorm] = LayerNorm(hidden_size)
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, ff_inner_dim, ff_inner_dim)

        self.wi: Optional[jnp.array] = random.normal(key, (hidden_size, sum(self.fused_dims)))
        self.attn_wo: Optional[jnp.array] = random.normal(key, (attn_inner_dim, hidden_size))
        self.ff_wo: Optional[jnp.array] = random.normal(key, (ff_inner_dim, hidden_size))

        self.n_heads: Optional[int] = n_heads
        self.scale: Optional[float] = dim_head ** -0.5
        self.mask_value: Union[int, float, Any] = mask_value

    def __call__(self, x, *, pos_emb, causal_mask):
        split_indices = onp.cumsum(self.fused_dims[:-1])

        x = self.norm(x)

        q, k, v, ff, ff_gate = np.split(x @ self.wi, split_indices, axis=-1)

        q = rearrange(q, '... n (h d) -> ... h n d', h=self.n_heads) * self.scale

        q, k = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k))

        sim = einsum('... h i d, ... j d -> ... h i j', q, k)

        attn = nn.softmax(np.where(causal_mask, sim, self.mask_value), axis=-1)

        out = einsum('... h i j, ... j d -> ... h i d', attn, v)

        attn_out = rearrange(out, '... h n hd -> ... n (h hd)') @ self.attn_wo

        ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

        return attn_out + ff_out


class PPaLM(Module):
    def __init__(self, config: Optional[PPaLMConfig]):
        hidden_size = config.hidden_size
        dim_head = config.dim_head
        n_layers = config.n_layers

        key = config.key

        self.embedding: np.ndarray = random.normal(key, (config.vocab_size, hidden_size)) * 0.02
        self.inv_freq: onp.ndarray = 1.0 / (10000 ** (np.arange(0, dim_head, 2) / dim_head))

        self.layers: List[PPaLMBlock] = [
            PPaLMBlock(config=config)
            for _
            in range(n_layers)
        ]
        self.norm: Module = LayerNorm(hidden_size)

    @jit
    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        rotary_emb = fixed_pos_embedding(self.inv_freq, n)
        causal_mask = np.tril(np.ones((n, n)))

        for block in self.layers:
            x = block(x, pos_emb=rotary_emb, causal_mask=causal_mask) + x

        x = self.norm(x)
        return x @ self.embedding.transpose()


def cross_entropy_loss(prediction: Union[jnp.arange, Any], targets: Union[jnp.arange, Any]):
    return -jnp.sum(prediction * targets)


if __name__ == "__main__":
    config = PPaLMConfig()
    model = PPaLM(config=config)
    seq = jax.random.randint(config.key, (1, 512), 0, config.vocab_size)
    logits = model(seq)
    print(logits)
