import os

import torch
from erutils.command_line_interface import fprint

from modules.commons import *
from utils.utils import GB


def train(data_path: [os.PathLike, str], epochs: int = 10000, lr: float = 4e-4, chunk_size: int = 728,
          pre_show_chunk: bool = False, batch_size: int = 10, set_seed: bool = False, seed: int = 1377,
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    if set_seed: torch.manual_seed(seed)
    with open(data_path, 'r') as stream:
        text = stream.read()
    split = int(0.9 * len(text))

    # attar_print(data_length=len(text))
    chars = sorted(list(set(text)))

    fprint(f'len Chars : {len(chars)}', end='\n')
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    fprint('Created String to integer Vocab ~ Successfully')
    i_to_s = {i: ch for i, ch in enumerate(chars)}
    fprint('Created integer to  String Vocab ~ Successfully')

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

    m = BLM(vocab_size=len(chars), chunk_size=chunk_size, n_embedded=324, head_size=64, n_layers=12, n_head=12)
    fprint('Generating a Poet with 100 length ...')
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    m = m.to(device)
    fprint(f'[[ Model Created with {sum(p.numel() for p in m.parameters()) / 1e6} M parameters Over All ]]')
    # v = m.generate(torch.zeros((1, 1), dtype=torch.long), 100)
    # fprint(decode(v[0].tolist()))
    optimizer = torch.optim.AdamW(m.parameters(), lr)
    for epoch in range(epochs):
        predict, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        fprint(f'\r Epoch [{epoch + 1}/{epochs}] | Loss : [{loss.item()}]', end='')
        if (epoch + 1) % 500 == 0:
            print()

            saves = {
                'model': m.state_dict(),
                'epochs': epochs,
                'lr': lr,
                'optim': optimizer.state_dict()
            }

            torch.save(saves, 'model.pt')

        if (epoch + 1) % 1000 == 0:
            fprint(f'Generating Some Samples To generated-{epoch}.txt')
            stream = open(f'generated-{epoch}.txt', 'w')
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            stream.write(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    stream = open('generated.txt', 'w')
    txt = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    print(txt)
    stream.write(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))

    # m = torch.jit.script(m)
    # torch.jit.save(m,
    #                'model.pt',
    #                # _extra_files=saves
    #                )
    # m.save('model.pt')
