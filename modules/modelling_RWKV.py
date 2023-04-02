import dataclasses

import torch
import math
from torch import nn
from typing import List, Dict, Tuple, Optional, Union
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RWKVConfig:
    number_of_layers: int = 8
    hidden_size: int = 512
    k_clamp: int = 60
    ctx_len: int = 128
    k_eps: float = 1e-8
    vocab_size: int = 32000
    eps: float = 1e-5
    device: str = 'cuda'


@dataclasses.dataclass
class RWKVConfigTrain:
    betas: Optional[Tuple[float]] = (0.90, 0.99)
    learning_rate: Optional[float] = 1e-4
    epochs: Optional[int] = 50
    eps: float = 0.1


class RWKV_CM(torch.jit.ScriptModule):
    def __init__(self, config: RWKVConfig, index):
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
        self.value = nn.Linear(h_up, hidden_size, bias=False)
        self.r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.value.scale_init = 0
        self.r.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        kv = self.value(F.silu(k))

        sr = F.sigmoid(self.r(xr))

        out = sr * kv
        return out


class RWKV_TM(torch.jit.ScriptModule):
    def __init__(self, config: RWKVConfig, index):
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
        h_up = hidden_size
        self.k = nn.Linear(hidden_size, h_up, bias=False)
        self.v = nn.Linear(hidden_size, h_up, bias=False)
        self.r = nn.Linear(hidden_size, h_up, bias=False)
        self.o = nn.Linear(h_up, hidden_size, bias=False)
        self.time_w = torch.tensor([0])
        self.k.scale_init = 0
        self.r.scale_init = 0
        self.o.scale_init = 0
        self.k_clamp = config.k_clamp
        self.k_eps = config.k_eps

    @torch.jit.script_method
    def wkv_run(self, k, v):
        B, T, C = k.shape

        time_decay = self.time_decay
        time_decay = - torch.exp(time_decay)
        carry = (torch.tensor([0.0] * C).to(time_decay.device), torch.tensor([0.0] * C).to(time_decay.device),
                 torch.tensor([-1e38] * C).to(time_decay.device))

        aa, bb, pp = carry
        # k = k.mT
        # v = v.mT
        ww = self.time_first + k
        # k = k.mT
        # v = v.mT
        # ww = ww.mT
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        # k = k.mT
        # v = v.mT
        ww = time_decay + pp
        # k = k.mT
        # v = v.mT
        # ww = ww.mT
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        aa = e1 * aa + e2 * v
        bb = e1 * bb + e2
        pp = p
        hidden_state = (aa, bb, pp)
        return a / b

    @torch.jit.script_method
    def func_jump(self, x):
        xx = self.time_shift(x)
        k = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        v = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        r = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.k(k)
        v = self.v(v)
        r = self.r(r)
        return r, k, v

    def forward(self, x):

        r, k, v = self.func_jump(x)
        wkv = self.wkv_run(k, v)
        sr = torch.sigmoid(r)
        rwkv = sr * wkv
        rwkv = self.o(rwkv)

        return rwkv


class RWKV_GPT_Block(nn.Module):
    def __init__(self, config, index: int):
        super(RWKV_GPT_Block, self).__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_size)
        self.post_norm = nn.LayerNorm(config.hidden_size)
        self.kwv = RWKV_TM(config=config, index=index)
        self.ffd = RWKV_CM(config=config, index=index)
        self.index = index

    def forward(self, x: torch.Tensor):
        copied_ln = self.pre_norm(x)
        x = x + self.kwv(copied_ln)
        copied_ln = self.post_norm(x)
        x = x + self.ffd(copied_ln)
        return x


class RWKV_Norm(torch.jit.ScriptModule):
    def __init__(self, config: RWKVConfig):
        super(RWKV_Norm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.eps

    @torch.jit.script_method
    def norm(self, x: torch.Tensor):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return self.weight * x


class RWKV_GPT_CasualLM(nn.Module):
    def __init__(self, config: RWKVConfig):
        super(RWKV_GPT_CasualLM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pre_ln = nn.LayerNorm(config.hidden_size)
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

    def forward(self, input_ids: torch.Tensor, target_ids=None):
        B, T = input_ids.size()
        hidden_state = self.embedding(input_ids)
        hidden_state = self.pre_ln(hidden_state)
        out = self.lm_head(self.post_ln(self.blocks(hidden_state)))
        loss = None
        if target_ids is not None:
            target_ids = target_ids[:, 1:].contiguous()
            out_p = out[:, :-1, :].contiguous()
            loss = F.cross_entropy(out_p.view(-1, out_p.size(-1)), target_ids.view(-1))
        return out, loss
