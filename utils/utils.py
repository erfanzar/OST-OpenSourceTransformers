import torch


class GB:
    def __init__(self, train_data, eval_data, batch_size, chunk_size):
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def forward(self, mode: str, *args, **kwargs):
        data = self.train_data if mode == 'train' else self.eval_data
        ix = torch.randint(len(data) - self.chunk_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.chunk_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.chunk_size + 1] for i in ix])
        return x, y


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)
