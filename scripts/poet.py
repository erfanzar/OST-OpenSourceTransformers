import os
import typing

import torch
from erutils.utils import read_yaml
from erutils.command_line_interface import fprint

from modules.models import PTTMultiHeadAttention


def poet(config_path: typing.Union[str, os.PathLike], path_model: typing.Union[str, os.PathLike],
         generate_token: int = 2000, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', ):
    cfg = read_yaml(config_path)
    data_path = cfg['data_path']
    chunk_size = cfg['chunk_size']
    number_of_head = cfg['number_of_head']
    use_compile = cfg['use_compile']
    number_of_layers = cfg['number_of_layers']
    head_size = cfg['head_size']
    number_of_embedded = cfg['number_of_embedded']

    for k, v in cfg.items():
        txt = f'\033[1;36m | \033[1;32m{k} : \033[1;36m{v}'
        print(txt, ' ' * abs(len(txt) - 100), '|')

    with open(data_path, 'r') as stream:
        text = stream.read()

    chars = sorted(list(set(text)))

    fprint(f'DEVICE : {device}')

    s_to_i = {ch: i for i, ch in enumerate(chars)}

    i_to_s = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [s_to_i[c] for c in s]
    decode = lambda l: ''.join([i_to_s[i] for i in l])

    m = PTTMultiHeadAttention(vocab_size=len(chars),
                              chunk_size=chunk_size,
                              number_of_embedded=number_of_embedded,
                              head_size=head_size,
                              number_of_layers=number_of_layers,
                              number_of_head=number_of_head).to(device)

    m_s = torch.load(path_model, map_location=device)

    fprint(f'Trained Learning Rate : {m_s["lr"]}')
    fprint(f'Targeted Epoch for train : {m_s["epochs"]}')
    fprint(f'Trained Epoch : {m_s["epoch"]}')
    m.load_state_dict(m_s['model'])
    if use_compile:
        fprint('Compiling Model For Speed Boost 🚀 ...')
        m = torch.compile(m)
        fprint('Model Compiled Successfully 🧠')
    txt = ''
    idx = torch.zeros(1, 1, dtype=torch.long if device == 'cuda' else torch.int).to(device)
    for i in range(generate_token):
        idx = m.generate(idx, 1)
        txt += decode(idx[0][-1])
        fprint(f'\r{txt}', end='')
    print('EXIT :)')
