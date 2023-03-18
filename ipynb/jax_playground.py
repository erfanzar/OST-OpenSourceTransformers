import dataclasses
import math

import jax
import jax.numpy as np
import time
from flax import linen as nn
from jax import jit, random
from tqdm.auto import tqdm
import logging
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
        logger.info(f'{func.__name__}  ==> {t2 - t1} sec')
        return ret

    return warper


@dataclasses.dataclass
class Config:
    n_heads: int = 8
    n_layers: int = 10
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


class LLMoFC(nn.Module):
    config: Config

    def setup(self) -> None:
        config: Config = self.config
        self.embedding = nn.Embed(config.vocab_size, config.hidden_size)
        self.blocks = [
            Block(config=config) for _ in range(config.n_layers)
        ]
        self.out = nn.Dense(config.vocab_size, **dense_kwargs)

    def __call__(self, input_ids, train=True, attention_mask=None):
        # print(f'praj : {praj}')
        hidden = self.embedding(input_ids).reshape(1, -1, self.config.hidden_size)
        # print(f'hidden : {hidden.shape}')
        for block in self.blocks:
            hidden = block(hidden, train=train, attention_mask=attention_mask)
        return self.out(hidden)


if __name__ == "__main__":
    hyper_parameters = Config()
    batch, seq_len, hidden_size = 1, hyper_parameters.max_sentence_length, hyper_parameters.hidden_size
    rk, rn = random.split(rk)
    dummpy_input = jax.random.normal(rn, (batch, seq_len)).astype(hyper_parameters.dtype_embedding)
    model = LLMoFC(hyper_parameters)
    rk, rn = jax.random.split(rk)
    params = model.init(rn, dummpy_input)
    rk, rn = random.split(rk)
    x = jax.random.normal(rn, (batch, seq_len)).astype(hyper_parameters.dtype_embedding)


    @timer
    @jit
    def run():
        out = model.apply(params, x)
        return True


    fake_data_size: int = 100
    seq_len_a: int = 8
    x = np.array(
        np.split(np.arange(seq_len_a * fake_data_size, dtype=hyper_parameters.dtype_embedding),
                 fake_data_size))
    y = np.array(
        np.split(np.arange(fake_data_size * seq_len_a, dtype=hyper_parameters.dtype_embedding),
                 fake_data_size))
    print(x.shape)
    print(y.shape)


    @jax.jit
    def cross_entropy(params, x_batched, y_batched):

        def ce(x, y):
            pred = model.apply(params, x)
            return -np.sum(pred * y)

        return jax.vmap(ce)(x_batched, y_batched)


    learning_rate: float = 2e-5
    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(cross_entropy)
    for i in range(10):
        with tqdm(zip(x, y), total=len(x)) as prb:
            for x_samples, y_samples in prb:
                loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
                updates, opt_state = tx.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                prb.set_postfix(loss=loss_val, epoch=i)
