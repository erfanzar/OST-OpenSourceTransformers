import deepspeed

from modules import LGeMConfig, LGeMForCausalLM
from utils.utils import count_model_parameters
import os
import multiprocessing as mp
from dataclasses import dataclass, field


@dataclass
class RunArgument:
    with_cuda: bool = field(default=False, metadata={
        'help': "train model using gpu"
    })
    use_ema: bool = field(default=False, metadata={
        'help': "whether use exponential moving average"
    })
    batch_size: int = field(default=8, metadata={
        'help': 'batch size to train model'
    })
    learning_rate: float = field(default=1e-4, metadata={
        'help': 'optimizer learning rate for training'
    })
    epochs: int = field(default=5, metadata={
        'help': 'number of training epochs'
    })

    local_rank: int = field(default=-1, metadata={
        'help': 'local rank passed from distributed launcher'
    })




