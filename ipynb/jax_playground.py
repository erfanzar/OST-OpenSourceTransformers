import dataclasses
import math

import jax
import jax.numpy as np
from flax import linen as nn
from jax import jit, random

# defice random key
seed: int = 42
rk = jax.random.PRNGKey(seed=seed)
dense_kwargs = dict(kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.zeros)


@dataclasses.dataclass
class Config:
    n_heads: int = 8
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

    def __call__(self, x: np.array, attention_mask=None):
        batch_size, seq_length, embed_dim = x.shape

        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.config.n_heads, self.head_dim * 3)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = np.array_split(qkv, 3, axis=-1)

        attention = scaled_dot_product(q, k, v, bias=self.bias, attention_mask=attention_mask) \
            .transpose(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)

        out = self.o_proj(attention)

        return out


class Block(nn.Module):
    config: Config

    def setup(self) -> None:
        self.attention = Attention(self.config)
        config = self.config
        self.mlp = [
            nn.Dense(config.hidden_size * 4, **dense_kwargs),
            nn.Dropout(config.drop_prob),
            nn.gelu,
            nn.Dense(config.hidden_size, **dense_kwargs)
        ]
        self.pre_ln = nn.RMSNorm()
        self.post_ln = nn.RMSNorm()

    def __call__(self, x: np.array, train=True, attention_mask=None):
        attn = self.attention(self.pre_ln(x), attention_mask=attention_mask)
        mlp_in = self.post_ln(x)
        for m in self.mlp:
            mlp_in = m(mlp_in) if isinstance(m, nn.Dense) else m(mlp_in, train)
        return x + mlp_in + attn


if __name__ == "__main__":
    hyper_parameters = Config()

    batch, seq_len, hidden_size = 12, 180, hyper_parameters.hidden_size
    rk, rn = random.split(rk)
    dummpy_input = jax.random.normal(rn, (batch, seq_len, hidden_size))

    model = Block(hyper_parameters)

    rk, rn = jax.random.split(rk)
    params = model.init(rn, dummpy_input)
    rk, rn = random.split(rk)
    x = jax.random.normal(rn, (batch, seq_len, hidden_size))
    out = model.apply(params, x)
    print(out.shape)
