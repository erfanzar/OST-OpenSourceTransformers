import os
import typing

import torch
from erutils.utils import read_yaml
from erutils.command_line_interface import fprint

from modules.models import PTTMultiHeadAttention


def load(config_path: typing.Union[str, os.PathLike], path_model: typing.Union[str, os.PathLike],
         generate_token: int = 2000, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', ):
    cfg = read_yaml(config_path)
    data_path = cfg['data_path']
    # epochs = cfg['epochs']
    lr = float(cfg['lr'])
    chunk_size = cfg['chunk_size']
    # pre_show_chunk = cfg['pre_show_chunk']
    # batch_size = cfg['batch_size']
    # set_seed = cfg['set_seed']
    # seed = cfg['seed']

    number_of_head = cfg['number_of_head']
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

    text = torch.tensor(encode(text), dtype=torch.long if device == 'cuda' else torch.int).to(device)
    m = PTTMultiHeadAttention(vocab_size=len('chars'), chunk_size=chunk_size, number_of_embedded=number_of_embedded,
                              head_size=head_size,
                              number_of_layers=number_of_layers,
                              number_of_head=number_of_head).to(device)

    m_s = torch.load(path_model, map_location=device)

    fprint(f'Trained Learning Rate : {m_s["lr"]}')
    fprint(f'Targeted Epoch for train : {m_s["epochs"]}')
    fprint(f'Trained Epoch : {m_s["epoch"]}')
    m.load_state_dict(m_s['model'])
    optimizer = torch.optim.AdamW(m.parameters(), m_s['lr'])
    optimizer.load_state_dict(m_s['optim'])
    if generate_token > 0:
        x = torch.ones((1, 1), dtype=torch.long if device == 'cuda' else torch.int).to(device)
        generated = m.generate(x, generate_token)[0].tolist()
        print(decode(generated))
    else:
        fprint('Skip Generating Text [Generate Token < 0]')
    return m, optimizer
