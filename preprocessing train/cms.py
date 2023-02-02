import torch


def add_special_tokens(text: str):
    assert isinstance(text, str), 'text input must be string '
    text = '[CLS]' + text + '[SEP]'
    return text


def add_pad(tensor, length: int = 512):
    t = torch.zeros(1, length)
    for i, c in enumerate(tensor.view(-1)):
        t[0, i] = c

    return t
