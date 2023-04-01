import torch
import math
from torch import nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)

n_embedding = 8
n_layers = 12


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

    @torch.jit.script_method
    def func_jump(self, x):
        xx = self.time_shift(x)
        k = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        v = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        r = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.k(k)
        v = self.v(v)
        sr = torch.sigmoid(self.r(r))
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()

        sr, k, v = self.jit_func(x)
        # TODO: implement run_cuda in pytorch
        # rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        # rwkv = self.output(rwkv)
        # return rwkv
        return ...
