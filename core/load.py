import os
import typing

import torch
from erutils.command_line_interface import fprint

from modules.commons import BLM


def load(path: typing.Union[str, os.PathLike], path_data: typing.Union[str, os.PathLike], vocab_size: int = 65,
         generate_token: int = 2000,
         chunk_size: int = 328, n_embedded: int = 324, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
         head_size: int = 64, n_layers: int = 12, n_head: int = 12):
    with open(path_data, 'r') as stream:
        text = stream.read()
    chars = sorted(list(set(text)))
    fprint(f'DEVICE : {device}')
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_s = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [s_to_i[c] for c in s]
    decode = lambda l: ''.join([i_to_s[i] for i in l])
    text = torch.tensor(encode(text), dtype=torch.long if device == 'cuda' else torch.int).to(device)
    m = BLM(vocab_size=vocab_size, chunk_size=chunk_size, n_embedded=n_embedded, head_size=head_size, n_layers=n_layers,
            n_head=n_head).to(device)

    m_s = torch.load(path, map_location=device)

    fprint(f'Trained Learning Rate : {m_s["lr"]}')
    fprint(f'Trained Epoch : {m_s["epochs"]}')
    m.load_state_dict(m_s['model'])
    x = torch.ones((1, 1), dtype=torch.long if device == 'cuda' else torch.int).to(device)
    generated = m.generate(x, generate_token)[0].tolist()
    print(generated)
    print(decode(generated))


if __name__ == "__main__":
    load('E:\\Programming\\Python\\Ai-Projects\\PTT-PoetryTrainerdTransformers\\model-AAA.pt',
         'E:\\Programming\\Python\\Ai-Projects\\PTT-PoetryTrainerdTransformers\\data\\input.txt')
