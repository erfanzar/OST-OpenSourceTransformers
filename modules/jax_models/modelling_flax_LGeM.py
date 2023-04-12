import jax
from jax import numpy as np
import flax.linen as nn
import optax
from jax import jit, vmap, pmap, grad
from jax.random import PRNGKey
from tqdm.auto import tqdm
import flax
import numpy as onp
import einops

jax.config.update('jax_platform_name', 'cpu')


class LGemConfig:
    def __init__(self,
                 initializer_range: float = 0.02,
                 hidden_size: int = 768,
                 dtype: np.dtype = np.float16,
                 intermediate_size: int = 2048,
                 num_hidden_layers: int = 4,
                 rms_norm_eps: int = 1e-6,
                 vocab_size: int = -1,
                 num_attention_heads: int = 8,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 weight_decay: float = 0.02,
                 learning_rate: float = 3e-4,
                 max_sequence_length: int = 768,
                 epochs: int = 500
                 , **kwargs):
        self.__dict__.update(**kwargs)
        self.initializer_range: float = initializer_range
        self.hidden_size: int = hidden_size
        self.dtype: np.dtype = dtype
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
        self.rms_norm_eps: int = rms_norm_eps
        self.vocab_size: int = vocab_size
        self.num_attention_heads: int = num_attention_heads
        self.use_cache: bool = use_cache
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id
        self.eos_token_id: int = eos_token_id
        self.weight_decay: float = weight_decay
        self.learning_rate: float = learning_rate
        self.max_sequence_length: int = max_sequence_length
        self.epochs: int = epochs

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


STANDARD_CONFIGS = {
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug': {
        'vocab_size': 182,
        'hidden_size': 256,
        'intermediate_size': 512,
        'num_hidden_layers': 8,
        'num_attention_heads': 4,
        'max_sequence_length': 512,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
        'dtype': np.float32
    },
}


def pre_compute_freq(dim, length=2048, theta: int = 10000, dtype=np.float16):
    freq = 1 / (theta ** (np.arange(0, dim, 2).astype(dtype) / dim))
    max_cache = np.arange(length).astype(dtype)
    freq = einops.einsum(max_cache, freq, 'i,j-> i j')
    freq = np.concatenate([freq, freq], axis=-1)
    sin = np.sin(freq).astype(dtype)[None, None, :, :]
    cos = np.cos(freq).astype(dtype)[None, None, :, :]
    return sin, cos


@jit
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_embedding(query, key, sin, cos):
    '''

    :param query: np.float16
    :param key: np.float16
    :param sin: np.float16
    :param cos: np.float16
    :return: query,key
    '''

    cos = cos[..., :key.shape[-2], :]
    sin = sin[..., :key.shape[-2], :]
    key = (key * cos) + (rotate_half(key) * sin)
    query = (query * cos) + (rotate_half(query) * sin)
    return query, key


class LGemAttention(nn.Module):
    config: LGemConfig

    def setup(self):
        dtype = self.config.dtype
        hidden_size = self.config.hidden_size

        num_attention_heads = self.config.num_attention_heads

        head_dims = hidden_size // num_attention_heads
        self.head_dims = head_dims
        self.num_attention_heads = num_attention_heads
        self.sin, self.cos = pre_compute_freq(head_dims, self.config.max_sequence_length * 2, dtype=dtype)
        self.q_proj = nn.Dense(head_dims * num_attention_heads, use_bias=False,
                               kernel_init=nn.initializers.normal(0.02), dtype=dtype)
        self.k_proj = nn.Dense(head_dims * num_attention_heads, use_bias=False,
                               kernel_init=nn.initializers.normal(0.02), dtype=dtype)
        self.v_proj = nn.Dense(head_dims * num_attention_heads, use_bias=False,
                               kernel_init=nn.initializers.normal(0.02), dtype=dtype)
        self.o_proj = nn.Dense(head_dims * num_attention_heads, use_bias=False,
                               kernel_init=nn.initializers.normal(0.02), dtype=dtype)
        self.drop = nn.Dropout(0.1)

    def merge_heads(self, x):
        x = x.reshape(x.shape[0], -1, self.head_dims * self.num_attention_heads)
        return x

    def split_heads(self, x):
        b, t, c = x.shape
        x = x.reshape(b, t, self.num_attention_heads, self.head_dims)
        return x

    def __call__(self, x, attention_mask, deterministic: bool = True):
        query = self.split_heads(self.q_proj(x))
        key = self.split_heads(self.k_proj(x))
        value = self.split_heads(self.v_proj(x))
        query, key = apply_rotary_embedding(query=query, key=key, cos=self.cos, sin=self.sin)
        depth = query.shape[-1]
        query = query / np.sqrt(depth).astype(self.config.dtype)
        attn_weights = np.einsum('...qhd,...khd->...hqk', query, key) + attention_mask
        attn_weights = jax.nn.softmax(attn_weights).astype(self.config.dtype)
        attn_weights = self.o_proj(self.merge_heads(np.einsum('...hqk,...khd->...qhd', attn_weights, value)))
        return self.drop(attn_weights, deterministic)


class LGemPMSNorm(nn.Module):
    config: LGemConfig
    eps: float = 1e-6
    dtype: np.dtype = np.float32
    param_dtype: np.dtype = np.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.config.hidden_size,),
            self.param_dtype,
        )

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return x * jax.lax.rsqrt(np.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        output = self._norm(x.astype(self.dtype)).astype(self.dtype)
        weight = np.asarray(self.weight, self.dtype)
        return output * weight


class LGemMLP(nn.Module):
    config: LGemConfig

    def setup(self):
        self.gate_act = nn.Dense(self.config.intermediate_size, use_bias=False)
        self.gate_non = nn.Dense(self.config.intermediate_size, use_bias=False)
        self.output = nn.Dense(self.config.hidden_size, use_bias=False)
        self.act = nn.silu
        self.drop = nn.Dropout(0.1)

    def __call__(self, x, deterministic: bool = True):
        return self.drop(self.output(self.act(self.gate_act(x)) * self.gate_non(x)), deterministic)


class LGemBlock(nn.Module):
    config: LGemConfig

    def setup(self):
        config = self.config
        self.pre_ln = LGemPMSNorm(config)
        self.post_ln = LGemPMSNorm(config)
        self.attn = LGemAttention(config)
        self.mlp = LGemMLP(config)

    def __call__(self, x, attention_mask, deterministic: bool = True):
        residual = x
        x = self.attn(self.pre_ln(x), attention_mask=attention_mask, deterministic=deterministic) + residual
        residual = x
        x = self.mlp(self.post_ln(x), deterministic=deterministic) + residual
        return x


class LGemModel(nn.Module):
    config: LGemConfig

    def setup(self):
        config = self.config
        self.embedding = nn.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, dtype=config.dtype)
        self.layers = [LGemBlock(config) for c in range(config.num_hidden_layers)]
        self.ln = LGemPMSNorm(config)

    def make_mask(self, x, attention_mask, ):
        assert x.ndim == 2
        b, t = x.shape
        if attention_mask is None:
            attention_mask = np.ones((b, t), self.config.dtype)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        mask = nn.combine_masks(attention_mask, nn.make_causal_mask(x)) - 1
        mask = np.where(mask == 0, 0, np.finfo(self.config.dtype).min)
        return mask

    def __call__(self, x, attention_mask=None, deterministic: bool = True):
        attention_mask = self.make_mask(x, attention_mask)
        hidden = self.embedding(x)
        for i, block in enumerate(self.layers):
            hidden = block(hidden, attention_mask=attention_mask, deterministic=deterministic)
        hidden = self.ln(hidden)
        return hidden


class LGemModelForCasualLM(nn.Module):
    config: LGemConfig

    def setup(self):
        config = self.config
        self.model = LGemModel(config)
        self.lm = nn.Dense(config.vocab_size, use_bias=False, dtype=config.dtype)

    def __call__(self, x, attention_mask=None, deterministic: bool = True):
        return self.lm(self.model(x, attention_mask=attention_mask, deterministic=deterministic))


def flax_count_params(params):
    _i = flax.core.unfreeze(params)
    _i = jax.tree_util.tree_flatten(_i)[0]
    return sum(i.size for i in _i)


def cross_entropy_loss(prediction: np.DeviceArray, targets: np.DeviceArray):
    targets = jax.nn.one_hot(targets, num_classes=prediction.shape[-1])
    prediction = nn.softmax(prediction)
    loss = - np.sum(np.log(prediction + 1e-12) * targets, axis=-1)
    loss = np.mean(loss)
    return loss
