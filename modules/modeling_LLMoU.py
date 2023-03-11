import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterable

import torch
from torch import Tensor
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class LLMoUConfig:
    dtype: Optional[torch.dtype] = torch.float32
    hidden_size: Optional[int] = 2048
    eps: Optional[float] = 1e-5
    n_heads: Optional[int] = 8
    n_layers: Optional[int] = 12
    use_cash: Optional[bool] = True
    epochs: Optional[int] = 100
    vocab_size: Optional[int] = -1
    max_sentence_length: Optional[int] = 512
    hidden_dropout: Optional[float] = 0.1
    training: Optional[bool] = True
    attention_dropout: Optional[float] = 0.1
    use_ln_for_residual: Optional[bool] = False
    weight_decay: Optional[float] = 2e-1
    initializer_range: Optional[float] = 0.02
    lr: Optional[float] = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_causal_mask(
        input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def build_alibi_tensor(attention_mask: torch.Tensor, n_heads: int, dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != n_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, n_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * n_heads, 1, seq_length).to(dtype)


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    logger.debug(f'Mask SHAPE  :  {mask.shape}')

    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


class LLMoUPMSNorm(nn.Module):
    def __init__(self, config: LLMoUConfig):
        super(LLMoUPMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size, dtype=config.dtype))
        self.eps = config.eps
        self.config = config

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x.dtype != self.config.dtype:
            x = x.type(self.config.dtype)
        hidden = self.norm(x)

        return self.weight * hidden


class LLMoUAttention(nn.Module):
    def __init__(self, config: LLMoUConfig):
        super(LLMoUAttention, self).__init__()
        self.n_heads = config.n_heads
        self.config = config
        self.head_dim = config.hidden_size // config.n_heads
        assert self.head_dim * config.n_heads == config.hidden_size
        self.k_q_v = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.alpha = torch.rsqrt(torch.tensor(self.head_dim))
        self.hidden_dropout = config.hidden_dropout

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.n_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.n_heads

        x = x.view(batch_size, self.n_heads, seq_length, self.head_dim)

        x = x.permute(0, 2, 1, 3)

        return x.reshape(batch_size, seq_length, self.n_heads * self.head_dim)

    def forward(self,
                input_ids: Optional[Tensor],
                attention_mask: Optional[Tensor],
                residual: Optional[Tensor],
                alibi: Optional[Tensor],
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                head_mask: Optional[torch.Tensor] = None,
                ):
        input_ids = input_ids.type_as(self.k_q_v.weight)

        kqv = self.k_q_v(input_ids)

        key, query, value = self._split_heads(kqv)
        batch, q_len, _, _ = query.shape
        key = key.permute(0, 2, 3, 1).view(batch * self.n_heads, self.head_dim, q_len)
        query = query.permute(0, 2, 1, 3).view(batch * self.n_heads, q_len, self.head_dim)
        value = value.permute(0, 2, 1, 3).view(batch * self.n_heads, q_len, self.head_dim)

        if layer_past is not None:
            key_c, val_c = layer_past
            key = torch.cat([key_c, key], dim=2)
            value = torch.cat([val_c, value], dim=1)

        _, _, kv_len = key.shape
        matmul_res = alibi.baddbmm(
            batch1=query,
            batch2=key,
            alpha=self.alpha,
            beta=1
        )

        attention_scores = matmul_res.view(batch, self.n_heads, q_len, kv_len)

        input_dtype = attention_scores.dtype

        attention_scores = attention_scores.to(torch.float) if input_dtype == torch.float16 else attention_scores
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs_reshaped = attention_probs.view(batch * self.n_heads, q_len, kv_len)

        context = self._merge_heads(torch.bmm(attention_probs_reshaped, value))

        output_tensor = self.dense(context)

        output_tensor = nn.functional.dropout(output_tensor, self.hidden_dropout, training=self.training) + residual

        return output_tensor


class LLMoUMLP(nn.Module):
    def __init__(self, config: LLMoUConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = nn.GELU()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = self.dropout(intermediate_output) + residual

        return output


class LLMoUBlock(nn.Module):
    def __init__(self, config: LLMoUConfig):
        super(LLMoUBlock, self).__init__()
        self.ln1 = LLMoUPMSNorm(config)
        self.self_attention: LLMoUAttention = LLMoUAttention(config)
        self.ln2 = LLMoUPMSNorm(config)
        self.mlp = LLMoUMLP(config)
        self.config = config

    def forward(self,
                hidden_state: Optional[Tensor],
                attention_mask: Optional[Tensor],
                alibi: Optional[Tensor],
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                head_mask: Optional[torch.Tensor] = None,

                ):
        layer_norm_output = self.ln1(hidden_state)
        residual = layer_norm_output if self.config.use_ln_for_residual else hidden_state
        attention_out = self.self_attention.forward(
            layer_norm_output,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            alibi=alibi,
            residual=residual
        )

        post_layer_norm = self.ln2(attention_out)

        residual = post_layer_norm if self.config.use_ln_for_residual else attention_out
        mlp_out = self.mlp(post_layer_norm, residual)

        return mlp_out


class LLMoUModel(nn.Module):
    def __init__(self, config: LLMoUConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.n_heads = config.n_heads
        self.config = config
        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LLMoUPMSNorm(config)
        self.dtype = config.dtype
        # Transformer blocks
        self.h = nn.ModuleList([LLMoUBlock(config) for _ in range(config.n_layers)])

        # Final Layer Norm
        self.ln_f = LLMoUPMSNorm(config)
        self.htw = nn.Linear(self.embed_dim, config.vocab_size)

        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LLMoUPMSNorm):
            module.weight.data.fill_(1.0)

    def _prepare_attn_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:

        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:

        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.LongTensor] = None,

            labels: Optional[torch.LongTensor] = None

    ) -> Union[Tuple[torch.Tensor, ...]]:

        batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )
        # from huggingface
        alibi = build_alibi_tensor(attention_mask=attention_mask, n_heads=self.n_heads, dtype=self.dtype)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            hidden_states = block(
                hidden_states,
                layer_past=layer_past,

                alibi=alibi,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
            )

        logits = self.htw(self.ln_f(hidden_states))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

    def generate(
            self,
            tokens: Optional[torch.Tensor],
            eos_id: int,
            attention_mask=None,
            max_gen_len: int = 20,
            temperature: float = 0.9,
            top_p: float = 0.95,
    ) -> Iterable[torch.Tensor]:
        def sample_top_p(probs, p):
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            _next_token = torch.multinomial(probs_sort, num_samples=1)

            _next_token = torch.gather(probs_idx, -1, _next_token)
            return _next_token

        for i in range(max_gen_len):
            tokens = tokens[:, -self.config.max_sentence_length:]
            logits, _ = self.forward(tokens, attention_mask)
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(*tokens.shape[:-1], 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.view(-1)[0] != eos_id:

                yield next_token.view(1, -1)
            else:
                break