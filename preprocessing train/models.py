import torch
import torch.nn as nn
from transformers import T5Model

class Conv1D(nn.Module):
    def __init__(self, c1, c2):
        super(Conv1D, self).__init__()
        w = torch.empty(c1, c2)
        torch.nn.init.normal_(w, std=0.02)
        self.c2 = c2
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(c2))

    def forward(self, x):
        out_shape = x.size()[:-1] + (self.c2,)
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).reshape(out_shape)


class Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(Attention, self).__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.scale_by_index = config.scale_attn_by_layer_idx
        self.num_heads = config.num_heads
        self.num_div = self.hidden_size // self.num_heads
        assert self.hidden_size % config.num_heads == 0
        self.c_attn = Conv1D(self.hidden_size, self.hidden_size * 3)
        self.c_proj = Conv1D(self.hidden_size, self.hidden_size)
        self.drop_out_attn = nn.Dropout(config.attn_dropout)
        self.drop_out_residual = nn.Dropout(config.residual_dropout)
        self.use_mask = config.use_mask
        self.register_buffer('mask',
                             torch.tril(
                                 torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                             .view(1, 1,
                                   config.max_position_embeddings,
                                   config.max_position_embeddings))

    def split_heads(self, tensor: torch.Tensor):
        new_shape = tensor.size()[:-1] + (self.num_heads, self.num_div)
        return tensor.reshape(new_shape).permute(0, 2, 1, 3)

    def merge_heads(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.size()[:-2] + (self.num_heads * self.num_div,)
        return tensor.reshape(new_shape)

    def attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask=None, head_mask=None):
        attn_weight = torch.matmul(query, key.transpose(-2, -1))
        attn_weight = attn_weight / torch.full([], value.size(-1) ** 0.5, dtype=attn_weight.dtype,
                                               device=attn_weight.device)
        if self.scale_by_index:
            attn_weight /= self.layer_idx

        if self.use_mask:
            key_len, query_len = key.size(-2), value.size(-2)
            masked = self.mask[:, :, key_len - query_len:query_len, :key_len].to(attn_weight.device)
            attn_weight = attn_weight.masked_fill(masked == 0, float('-inf'))

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weight = attn_weight + attention_mask

        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attn_weight = self.drop_out_attn(attn_weight)
        attn_weight = attn_weight.type(value.dtype)

        if head_mask is not None:
            attn_weight = attn_weight * head_mask
        attn_weight = torch.matmul(attn_weight, value)
        return attn_weight

    def forward(self, hidden_state, attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_state).split(self.hidden_size, 2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        attn = self.attn(query=query, key=key, value=value, attention_mask=attention_mask, head_mask=head_mask)
        return self.drop_out_residual(self.c_proj(self.merge_heads(attn)))


