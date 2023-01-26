import os
import typing

from erutils.utils import read_yaml
import torch
from erutils.command_line_interface import fprint

from modules.commons import *
from utils.utils import GB


def train(config_path: typing.Union[str, os.PathLike],
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    cfg = read_yaml(config_path)
    data_path = cfg['data_path']
    epochs = cfg['epochs']
    lr = float(cfg['lr'])
    chunk_size = cfg['chunk_size']
    pre_show_chunk = cfg['pre_show_chunk']
    batch_size = cfg['batch_size']
    set_seed = cfg['set_seed']
    seed = cfg['seed']

    number_of_head = cfg['number_of_head']
    number_of_layers = cfg['number_of_layers']
    head_size = cfg['head_size']
    number_of_embedded = cfg['number_of_embedded']

    for k, v in cfg.items():
        txt = f'\033[1;32m | {k} : \033[1;36m{v}'
        print(txt, ' ' * abs(len(txt) - 100), '|')

    if set_seed: torch.manual_seed(seed)
    with open(data_path, 'r') as stream:
        text = stream.read()
    split = int(0.9 * len(text))

    # attar_print(data_length=len(text))
    chars = sorted(list(set(text)))

    fprint(f'len Chars : {len(chars)}\n', end='\n')
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    fprint('Created String to integer Vocab ~ Successfully !!\n')
    i_to_s = {i: ch for i, ch in enumerate(chars)}
    fprint('Created integer to String Vocab ~ Successfully  !!\n')

    # encode = lambda s: [i_to_s[c] for c in s]
    encode = lambda s: [s_to_i[c] for c in s]

    # decode = lambda l: ''.join([s_to_i[c] for c in l])
    decode = lambda l: ''.join([i_to_s[i] for i in l])
    text = torch.tensor(encode(text), dtype=torch.long)

    train_data = text[:split]
    valid_data = text[split:]
    ptt_text = 'Wellcome To PTT or Poetry Trained Transformer'
    if pre_show_chunk:
        fprint(f'Example for word [{ptt_text}]')
        fprint(encode(ptt_text))
        fprint(decode(encode(ptt_text)))
    modes = ['train', "valid"]
    if pre_show_chunk:
        train_chunk_x = text[:chunk_size]
        train_chunk_y = text[1:chunk_size + 1]

        for t in range(chunk_size):
            x = train_chunk_x[:t + 1]
            y = train_chunk_y[t]

            fprint(
                f'Decoded \nInput : {x} | target : {y}\nEncoded\nInput : {encode(x)} | target : {encode(y)}\n---------------')
    else:
        fprint(f'[SKIP] PreShow Status is OFF ! ')

    get_batch = GB(train_data=train_data, valid_data=valid_data, batch_size=batch_size, chunk_size=chunk_size)

    # xb, yb = get_batch('train')
    # for b in range(batch_size):
    #     for t in range(chunk_size):
    #         context = xb[b, :t + 1]
    #         target = yb[b, t]
    #         print(f"when input is {context.tolist()} the target: {target}")

    m = BLM(vocab_size=len(chars), chunk_size=chunk_size, number_of_embedded=number_of_embedded, head_size=head_size,
            number_of_layers=number_of_layers,
            number_of_head=number_of_head)
    # fprint('Generating a Poet with 100 length ...')

    m = m.to(device)
    fprint(f'[[ Model Created with {sum(p.numel() for p in m.parameters()) / 1e6} M parameters Over All ]]',
           color='\033[1;32m')
    # v = m.generate(torch.zeros((1, 1), dtype=torch.long), 100)
    # fprint(decode(v[0].tolist()))
    optimizer = torch.optim.AdamW(m.parameters(), lr)
    last_valid_loss = 'NONE'
    for epoch in range(epochs):
        for mode in modes:
            x, y = get_batch('train')
            x, y = x.to(device), y.to(device)
            predict, loss = m(x, y)
            if mode not in ['valid', 'test']:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                last_valid_loss = loss.item()
            fprint(
                f'\rEpoch [{epoch + 1}/{epochs}] | Loss : [{loss.item()}] | Mode : [{mode}] | Last Validation Loss : [{last_valid_loss}]',
                end='')
            if (epoch + 1) % 500 == 0:
                print()

                saves = {
                    'model': m.state_dict(),
                    'epochs': epochs,
                    'epoch': epoch + 1,
                    'lr': lr,
                    'optim': optimizer.state_dict()
                }

                torch.save(saves, 'model.pt')

            if (epoch + 1) % 1000 == 0:
                fprint(f'Generating Some Samples To generated-{epoch + 1}.txt')
                stream = open(f'generated-{epoch + 1}.txt', 'w')
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                stream.write(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    stream = open('generated.txt', 'w')
    txt = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    stream.write(txt)
