import time
from torch.distributed import is_initialized
import torch
import os

try:
    import wandb
except ModuleNotFoundError:
    pass

LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))


class Timer:

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        started_ = self.started_
        if self.started_:
            self.stop()
        elapsed_ = self.elapsed_
        if reset:
            self.reset()
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):

        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True, print_rank_0_only=False):
        assert normalizer > 0.0
        if print_rank_0_only and LOCAL_RANK == 0:
            string = "time (ms)"
            if isinstance(names, str):
                names = [names]
            for name in names:
                elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
                string += " | {}  [RANK : {} ]:  {:.2f}".format(name, LOCAL_RANK, elapsed_time)
            if is_initialized():
                if LOCAL_RANK == 0:
                    print(string, flush=True)
            else:
                print(string, flush=True)
        else:
            string = "time (ms)"
            if isinstance(names, str):
                names = [names]
            for name in names:
                elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
                string += " | {}  [RANK : {} / {}]:  {:.2f}".format(name, LOCAL_RANK + 1, WORLD_SIZE, elapsed_time)
            if is_initialized():
                if LOCAL_RANK == 0:
                    print(string, flush=True)
            else:
                print(string, flush=True)
