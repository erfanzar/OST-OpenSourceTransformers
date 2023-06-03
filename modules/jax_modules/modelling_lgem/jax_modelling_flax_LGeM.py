import math

import flax.linen.partitioning
import jax
from flax.traverse_util import unflatten_dict, flatten_dict
from jax.sharding import PartitionSpec
import numpy as np
from jax.experimental.pjit import with_sharding_constraint as wsc
from flax.core import freeze, unfreeze
from flax import linen as nn

from jax import numpy as jnp
from transformers import PretrainedConfig, FlaxPreTrainedModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from functools import partial
from typing import Optional, Union

from einops import rearrange
from jax.interpreters import pxla


class FlaxLGeMConfig(PretrainedConfig):
    # HuggingFace FlaxLGeMConfig
    model_type = "LGeM"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            fsdp=False,
            gradient_checkpointing='',
            embedding_dropout: float = 0.,
            fcm_min_ratio: float = 0.,
            fcm_max_ratio: float = 0.,
            residual_dropout: float = 0.,
            attention_dropout: float = 0.,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.fsdp = fsdp
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_max_ratio = fcm_max_ratio
        self.fcm_min_ratio = fcm_min_ratio
        self.embedding_dropout = embedding_dropout
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def get_partition_rules():
        return (
            # Emb
            ("model/embed_tokens/embedding", PartitionSpec("mp", "fsdp")),

            # ATTn
            ("self_attn/(k_proj|v_proj|q_proj)/kernel", PartitionSpec("fsdp", "mp")),
            ("self_attn/o_proj/kernel", PartitionSpec("mp", "fsdp")),

            # MLP
            ("mlp/down_proj/kernel", PartitionSpec("mp", "fsdp")),
            ("mlp/up_proj/kernel", PartitionSpec("fsdp", "mp")),
            ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "mp")),

            # Norms
            ('norm/kernel', PartitionSpec(None)),
            ('input_layernorm/kernel', PartitionSpec(None)),
            ('post_attention_layernorm/kernel', PartitionSpec(None)),

            # OUT
            ("lm_head/kernel", PartitionSpec("fsdp", "mp")),
            ('.*', PartitionSpec(None)),

        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')


ACT2CLS = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu6": nn.relu6,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "swish": nn.swish,
    "tanh": nn.tanh,
}


def compute_freq_d(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def compute_freq(dim: int, man_length: int, theta: int = 10000):
    freq = 1 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(man_length)
    m = jnp.einsum('i,j->ij', t, freq)
    m = jnp.concatenate([m, m], axis=-1)
    cos = jnp.cos(m)
    sin = jnp.sin(m)
    return cos, sin


def apply_rotary_embedding___d(
        q: jnp.ndarray,
        k: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        dtype: jnp.dtype = jnp.float32,
):
    reshape_xq = q.astype(jnp.float32).reshape(*q.shape[:-1], -1, 2)
    reshape_xk = k.astype(jnp.float32).reshape(*k.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, xq_.shape)
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(dtype), xk_out.astype(dtype)


def rotate_half(tensor):
    depth = tensor.shape[-1]
    x1 = tensor[..., :depth]
    x2 = tensor[..., depth:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_embedding(q, k, c, s):
    b, h, l, d = q.shape
    c = c[0, 0, :l, :]
    s = s[0, 0, :l, :]
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    return q, k


def get_names_from_parition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint_(x, partition_specs):
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


class PMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = jnp.asarray(self.weight, self.dtype)
        variance = jnp.power(2, x).mean(-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return weight * x


class RotaryEmbedding(nn.Module):
    config: FlaxLGeMConfig

    def setup(self) -> None:
        dim = self.config.hidden_size // self.config.num_attention_heads
        self.cos, self.sin = compute_freq(dim, self.config.max_position_embeddings)
        self.dim = dim

    def __call__(self, x, max_l):
        if self.sin.shape[0] < max_l:
            self.cos, self.sin = compute_freq(self.dim, max_l)
        return self.cos[jnp.newaxis, jnp.newaxis, :, :].astype(x.dtype), self.sin[jnp.newaxis, jnp.newaxis, :,
                                                                         :].astype(x.dtype)


class LGeMSelfAttention(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        dense = partial(nn.Dense,
                        features=self.config.hidden_size,
                        kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                        use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
                        )
        self.k_proj = dense()
        self.v_proj = dense()
        self.q_proj = dense()
        self.o_proj = dense()
        self.rotary_embedding = RotaryEmbedding(config=self.config)
        self.scale = self.config.hidden_size // self.config.num_attention_heads

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray = None
                 ):
        b, t, c = hidden_states.shape

        k = self.k_proj(hidden_states)
        q = self.q_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.config.fsdp:
            q = with_sharding_constraint_(q, PartitionSpec(("dp", "fsdp"), None, "mp"))
            k = with_sharding_constraint_(k, PartitionSpec(("dp", "fsdp"), None, "mp"))
            v = with_sharding_constraint_(v, PartitionSpec(("dp", "fsdp"), None, "mp"))

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.config.num_attention_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.config.num_attention_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.config.num_attention_heads)
        cos, sin = self.rotary_embedding(x=k, max_l=t)

        k, q = apply_rotary_embedding(k=k, q=q, c=cos, s=sin)
        k = rearrange(k, 'b h s d -> b h d s')
        attn = jnp.matmul(q, k) / math.sqrt(self.scale)

        if attention_mask is not None:
            attn += attention_mask

        if self.config.fsdp:
            attn = with_sharding_constraint_(attn, PartitionSpec(("dp", "fsdp"), "mp", None, None))
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(attn.dtype)
        attn = jnp.matmul(attn, v).swapaxes(1, 2)
        attn = self.o_proj(attn.reshape(b, t, c))

        return attn


class LGeMMLP(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        dense = partial(nn.Dense,

                        use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype,
                        kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                        precision=self.precision
                        )

        self.gate_proj = dense(self.config.intermediate_size)
        self.up_proj = dense(self.config.intermediate_size)
        self.down_proj = dense(self.config.hidden_size)
        self.act = ACT2CLS[self.config.hidden_act]

    def __call__(self, x, ):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class LGeMBlock(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.self_attn = LGeMSelfAttention(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.mlp = LGeMMLP(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.input_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.post_attention_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps,
                                                dtype=self.dtype)

    def __call__(self,
                 hidden_state: jnp.DeviceArray,
                 attention_mask: jnp.DeviceArray = None
                 ):
        hidden_state = self.self_attn(self.input_layernorm(hidden_state),
                                      attention_mask=attention_mask,

                                      ) + hidden_state

        return self.mlp(self.post_attention_layernorm(hidden_state)) + hidden_state


class FlaxLGeMPretrainedModel(FlaxPreTrainedModel):
    module_class: nn.Module = None
    module_config: FlaxLGeMConfig
    base_model_prefix: str = 'model'

    def __init__(
            self,
            config: FlaxLGeMConfig,
            input_shape=(1, 256),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = False,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape, params=None):

        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            return_dict=False,
        )

        random_params = module_init_outputs["params"]
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
            self,
            input_ids: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            params: dict = None,
            return_dict: Optional[bool] = None,
            add_params_field: bool = False,
    ):

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        inputs = {'params': params or self.params} if add_params_field else params or {'params': self.params}

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4") if attention_mask is not None else attention_mask,
            return_dict=return_dict,

        )

        return outputs


def get_gradient(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


class FlaxLGeMCollection(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        block = LGeMBlock
        if self.config.gradient_checkpointing != '':
            LGeMCheckPointBlock = flax.linen.partitioning.remat(
                block,
                policy=get_gradient(self.config.gradient_checkpointing)
            )
            block = LGeMCheckPointBlock
        self.blocks = [
            block(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
            for _ in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_state: jnp.DeviceArray,
            attention_mask: jnp.DeviceArray = None,
            return_dict: bool = True,
    ):
        hidden_states = []

        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                attention_mask=attention_mask,

            )
            hidden_states.append(hidden_state)

        return hidden_state, hidden_states


class FlaxLGeMModule(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.wte = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        self.block = FlaxLGeMCollection(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.norm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def __call__(self,
                 input_ids,
                 attention_mask=None,
                 return_dict: bool = True,
                 ):
        b, s = input_ids.shape
        last_hidden_state = self.wte(input_ids)

        if attention_mask is None:
            attention_mask = jnp.ones((b, 1, 1, s))
        attention_mask = attention_mask.reshape(b, 1, 1, s)
        min_v = jnp.finfo(last_hidden_state.dtype).min
        attention_mask = jnp.where(nn.make_causal_mask(input_ids) == 0, min_v, 0) + jnp.where(attention_mask > 0, 0,
                                                                                              min_v)

        last_hidden_state, hidden_states = self.block(
            hidden_state=last_hidden_state,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        last_hidden_state = self.norm(last_hidden_state)

        if return_dict:
            return FlaxBaseModelOutput(
                hidden_states=hidden_states,
                last_hidden_state=last_hidden_state
            )
        else:
            return last_hidden_state


class FlaxLGeMModel(FlaxLGeMPretrainedModel):
    module_class = FlaxLGeMModule


class FlaxLGeMForCausalLMModule(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = FlaxLGeMModule(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.lm_head = nn.Dense(features=self.config.vocab_size, use_bias=False, dtype=self.dtype,
                                param_dtype=self.param_dtype,
                                kernel_init=nn.initializers.normal(self.config.initializer_range),
                                )

    def __call__(self,
                 input_ids,
                 attention_mask=None,
                 return_dict: bool = True,
                 ):

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        hidden_state = output.last_hidden_state if return_dict else output
        hidden_states = output.hidden_states if return_dict else None
        pred = self.lm_head(hidden_state)
        if return_dict:
            return FlaxCausalLMOutput(
                logits=pred,
                hidden_states=hidden_states
            )
        else:
            return pred,


class FlaxLGeMForCausalLM(FlaxLGeMPretrainedModel):
    module_class = FlaxLGeMForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, attention_mask: Optional[jnp.DeviceArray] = None,
                                      ):
        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,

        }
