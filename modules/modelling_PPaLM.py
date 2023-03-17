import jax.numpy as jnp
from jax import grad, vmap, pmap, device_count, device_get, random
from typing import Union, Optional
import numpy as np


def attention(key: Optional[jnp.array],
              query: Optional[jnp.array],
              value: Optional[jnp.array],
              attention_mask: Optional[jnp.array] = None,
              bias: Optional[jnp.array] = None):
    ...


if __name__ == "__main__":
    seq_len = 16
    head_dim = 100
    x = np.random.rand((1, seq_len, 8, head_dim * 3))
    key = x[..., :head_dim]
    query = x[..., head_dim:head_dim * 2]
    value = x[..., head_dim * 2:]
    attention_mask = np.ones(seq_len)
    attn = attention(key=key, query=query, value=value, attention_mask=attention_mask)
