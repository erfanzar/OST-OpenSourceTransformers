from functools import partial
from typing import Optional, Union

import flax.linen.partitioning
import jax
from einops import rearrange, repeat
from flax import linen as nn
from jax import numpy as jnp
from jax.experimental.pjit import with_sharding_constraint as wsc
from jax.interpreters import pxla
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig, FlaxPreTrainedModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput


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
            hidden_act="gelu",
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
            ("model/wte/embedding", PartitionSpec("mp", "fsdp")),

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


def fixed_pos_embedding(inv_freq, seq):
    sinusoid_inp = jnp.einsum('i , j -> i j', jnp.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, '... d -> ... (d r)', r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos[jnp.newaxis, :, jnp.newaxis, :]) + (rotate_every_two(x) * sin[jnp.newaxis, :, jnp.newaxis, :])


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

        self.scale = self.config.hidden_size // self.config.num_attention_heads

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 pos_emb,
                 attention_mask: jnp.DeviceArray = None,

                 ):
        b, t, c = hidden_states.shape

        k = self.k_proj(hidden_states)
        q = self.q_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.config.fsdp:
            q = with_sharding_constraint_(q, PartitionSpec(("dp", "fsdp"), None, "mp"))
            k = with_sharding_constraint_(k, PartitionSpec(("dp", "fsdp"), None, "mp"))
            v = with_sharding_constraint_(v, PartitionSpec(("dp", "fsdp"), None, "mp"))

        q = rearrange(q, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.config.num_attention_heads)

        k, q = map(lambda f: apply_rotary_pos_emb(f, pos_emb), (k, q))

        attn = nn.attention.dot_product_attention_weights(
            query=q,
            key=k,
            dtype=self.dtype,
            precision=self.precision,
            deterministic=False,
            bias=attention_mask
        )

        if self.config.fsdp:
            attn = with_sharding_constraint_(attn, PartitionSpec(("dp", "fsdp"), "mp", None, None))

        attn = jnp.einsum("...hqk,...khd->...qhd", attn, v)
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
        self.down_proj = dense(self.config.hidden_size)
        self.act = ACT2CLS[self.config.hidden_act]

    def __call__(self, x, ):
        return self.down_proj(self.act(self.gate_proj(x)))


class LGeMBlock(nn.Module):
    config: FlaxLGeMConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.self_attn = LGeMSelfAttention(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.mlp = LGeMMLP(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        # self.input_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        # self.post_attention_layernorm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps,
        #                                         dtype=self.dtype)

        self.input_layernorm = nn.LayerNorm(use_bias=False)
        self.post_attention_layernorm = nn.LayerNorm(use_bias=False)

    def __call__(self,
                 hidden_state: jnp.DeviceArray,
                 pos_emb: tuple,
                 attention_mask: jnp.DeviceArray = None
                 ):
        hidden_state = self.self_attn(self.input_layernorm(hidden_state),
                                      attention_mask=attention_mask,
                                      pos_emb=pos_emb,
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

        if params is None:
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                return_dict=False,
            )
            return module_init_outputs

        else:
            return params

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
            pos_emb: tuple,
            attention_mask: jnp.DeviceArray = None,
            return_dict: bool = True,
    ):
        hidden_states = []

        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                pos_emb=pos_emb
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
        # self.norm = PMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.norm = nn.LayerNorm(use_bias=False)
        h_d = self.config.hidden_size // self.config.num_attention_heads
        self.freq = 1 / (10000 ** (jnp.arange(0, h_d, 2) / h_d))

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

        attention_mask = jnp.clip(attention_mask, min_v, 0)
        pos_emb = fixed_pos_embedding(self.freq, s)
        last_hidden_state, hidden_states = self.block(
            hidden_state=last_hidden_state,
            attention_mask=attention_mask,
            return_dict=return_dict,
            pos_emb=pos_emb
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
