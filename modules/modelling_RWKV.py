import dataclasses

import torch
import math
from torch import nn
from typing import List, Dict, Tuple, Optional, Union
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


class RWKV_CM(torch.jit.ScriptModule):
    def __init__(self, config, index):
        super(RWKV_CM, self).__init__()
        layers = config.number_of_layers
        hidden_size = config.hidden_size
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            time_ratio = 1 - (index / layers)
            x = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                x[0, 0, i] = i / hidden_size
            self.time_mix_k = nn.Parameter(torch.pow(x, time_ratio))
            self.time_mix_r = nn.Parameter(torch.pow(x, time_ratio))

        h_up = hidden_size * 4

        self.key = nn.Linear(hidden_size, h_up, bias=False)
        self.value = nn.Linear(h_up, h_up, bias=False)
        self.r = nn.Linear(h_up, hidden_size, bias=False)

        self.value.scale_init = 0
        self.r.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        kv = self.value(F.silu(k))
        out = F.sigmoid(self.r(xr)) + kv
        return out


class RWKV_TM(torch.jit.ScriptModule):
    def __init__(self, config, index):
        super(RWKV_TM, self).__init__()
        layers = config.number_of_layers
        hidden_size = config.hidden_size
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            time_ratio_0_to_1 = (index / (layers - 1))
            time_ratio_1_to_pz = (1 - (index / layers))
            decay_speed = torch.ones(hidden_size)
            for i in range(hidden_size):
                decay_speed[i] = -5 + 8 * (i / (hidden_size - 1)) ** (0.7 + 1.3 * time_ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(hidden_size)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(hidden_size) * math.log(0.3) + zigzag)
            x = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                x[0, 0, i] = i / hidden_size
            self.time_mix_k = nn.Parameter(torch.pow(x, time_ratio_1_to_pz))
            self.time_mix_v = nn.Parameter(torch.pow(x, time_ratio_1_to_pz) + 0.3 * time_ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * time_ratio_1_to_pz))
        h_up = hidden_size * 4
        self.k = nn.Linear(hidden_size, h_up, bias=False)
        self.v = nn.Linear(hidden_size, h_up, bias=False)
        self.r = nn.Linear(hidden_size, h_up, bias=False)
        self.o = nn.Linear(h_up, hidden_size, bias=False)
        self.k.scale_init = 0
        self.r.scale_init = 0
        self.o.scale_init = 0
        self.k_clamp = config.k_clamp
        self.k_eps = config.k_eps

    @torch.jit.script_method
    def func_jump(self, x):
        xx = self.time_shift(x)
        k = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        v = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        r = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.k(k).transpose(-1, -2)
        v = self.v(v).transpose(-1, -2)
        r = self.r(r)
        k = torch.exp(torch.clamp(k, max=self.k_clamp))
        kv = k * v
        return r, k, v, kv

    def forward(self, x):
        B, T, C = x.size()

        r, k, v, kv = self.jit_func(x)
        self.time_w = torch.cat([torch.exp(self.time_decay), self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        w = w[:, -T:].unsqueeze(1)
        wkv = F.conv1d(nn.ZeroPad2d((T - 1, 0, 0, 0))(kv), w, groups=C)
        wk = F.conv1d(nn.ZeroPad2d((T - 1, 0, 0, 0))(k), w, groups=C) + self.k_eps

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)

        rwkv = self.output(rwkv)
        return rwkv


@dataclasses.dataclass
class RWKVConfig:
    number_of_layers: int = 8
    hidden_size: int = 512
    k_clamp: int = 60
    k_eps: float = 1e-8
    vocab_size: int = 32000
    eps: float = 1e-5


@dataclasses.dataclass
class RWKVConfigTrain:
    betas: Optional[Tuple[float]] = (0.90, 0.99)
    lr: Optional[float] = 1e-4
    epochs: Optional[int] = 50


class RWKV_GPT_Block(nn.Module):
    def __init__(self, config, index: int):
        super(RWKV_GPT_Block, self).__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_size)
        self.post_norm = nn.LayerNorm(config.hidden_size)
        self.kwv = RWKV_TM(config=config, index=index)
        self.ffd = RWKV_CM(config=config, index=index)
        self.index = index
        if self.index == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        if self.index == 0:
            x = self.ln0(x)
        x = x + self.kwv(self.pre_norm(x))
        x = x + self.ffd(self.post_norm(x))
        return x


class RWKV_Norm(torch.jit.ScriptModule):
    def __init__(self, config: RWKVConfig):
        super(RWKV_Norm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.eps

    @torch.jit.script_method
    def norm(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x

    def forward(self, x):
        x = self.norm(x)
        return self.weight * x


class RWKV_GPT_CasualLM(nn.Module):
    def __init__(self, config: RWKVConfig):
        super(RWKV_GPT_CasualLM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.Sequential(*(RWKV_GPT_Block(config=config, index=i) for i in range(config.number_of_layers)))
        self.post_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-5)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas,
                                     eps=train_config.eps)

        return optimizer

    def forward(self, input_ids, target_ids=None):
        B, T = input_ids.size()
        hidden_state = self.embedding(input_ids)
        out = self.lm_head(self.post_ln(self.blocks(hidden_state)))
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), target_ids.view(-1))
        return out, loss
