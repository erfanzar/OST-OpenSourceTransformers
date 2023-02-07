import os
import typing

from erutils.utils import read_yaml
import torch
from erutils.command_line_interface import fprint
import torch
from datasets import load_dataset
from torch.nn import functional as F
from utils.utils import DatasetQA, save_model
from torch.utils.data import DataLoader
from modules.models import PTT
from torch.utils.tensorboard import SummaryWriter


def train(config_path, device):
    ssm = SummaryWriter(log_dir='./out')

    max_length: int = 56
    embedded: int = 256
    number_of_heads: int = 4
    number_of_layers: int = 6
    # dataset = DatasetQA(max_length=max_length)

    squad_dataset = load_dataset('squad')
    train_data = squad_dataset['train']
    data_len = train_data.num_rows
    questions = train_data.data['question']
    answers = train_data.data['answers']
    dataset = DatasetQA(max_length=max_length, src=questions[:40], trg=answers[:40])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    vocab_size: int = dataset.vocab_size

    pad_index: int = dataset.pad_token_id

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ptt = PTT(
        vocab_size=vocab_size,
        max_length=max_length,
        embedded=embedded,
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        pad_index=pad_index
    ).to(device)
    print(sum(s.numel() for s in ptt.parameters()) / 1e6, " Million Parameters Are In MODEL")
    optimizer = torch.optim.AdamW(ptt.parameters(), 4e-4)
    epochs = 400
    # print(dataset.__getitem__(5))
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device).squeeze(0)
            y = y.to(device).squeeze(0)

            # print(f'X SHAPE : {x.shape} "|" Y SHAPE : {y.shape}')
            trg = y[:, 1:]
            #
            ys = y[:, :-1].contiguous().view(-1)

            # print(f'TARGET : {trg} "|" Y : {ys}')

            predict = ptt(x, trg)
            optimizer.zero_grad()
            loss = F.cross_entropy(predict.view(-1, predict.size(-1)), ys, ignore_index=pad_index)

            # print(predict_sa.shape)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f'\033[1;36m\r[{epoch + 1}/{epochs}] | Loss : {loss.item()} | Iter : {i + 1} | epoch_loss : {epoch_loss / (i + 1)}',
                end='')
            if (i + 1) % 20 == 0:
                # example_question = dataset.decode(x[0])
                # example_answer = dataset.decode(y[0])
                example_question = dataset.decode(x)
                example_answer = dataset.decode(y)
                # print(predict.shape)
                predict_sa = torch.multinomial(torch.softmax(predict[0], dim=-1), num_samples=1).view(1, -1)
                prra = dataset.decode(predict_sa)
                ssm.add_text('QUESTION', example_question, i + 1)
                ssm.add_text('ANSWER', example_answer, i + 1)

                ssm.add_text('PREDICT', prra, i + 1)
                ssm.add_scalar('train/LOSS', loss.item(), i + 1)
            # dataset.brea()

        print('\n')
        if epoch % 10 == 0:
            save_model(model=ptt.state_dict(), optimizer=optimizer.state_dict(), epochs=epochs, epoch=epoch,
                       name='TDFA.pt')

# def train(config_path: typing.Union[str, os.PathLike],
#           device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#     cfg = read_yaml(config_path)
#     data_path = cfg['data_path']
#     epochs = cfg['epochs']
#     lr = float(cfg['lr'])
#     chunk_size = cfg['chunk_size']
#     pre_show_chunk = cfg['pre_show_chunk']
#     batch_size = cfg['batch_size']
#     set_seed = cfg['set_seed']
#     seed = cfg['seed']
#     use_compile = cfg['use_compile']
#     number_of_head = cfg['number_of_head']
#     number_of_layers = cfg['number_of_layers']
#     head_size = cfg['head_size']
#     number_of_embedded = cfg['number_of_embedded']
#
#     load_weights = cfg['load_weights']
#     path_weights = cfg['path_weights']
#     for k, v in cfg.items():
#         txt = f'\033[1;36m | \033[1;32m{k} : \033[1;36m{v}'
#         print(txt, ' ' * abs(len(txt) - 100), '|')
#
#     if set_seed: torch.manual_seed(seed)
#     with open(data_path, 'r') as stream:
#         text = stream.read()
#     split = int(0.9 * len(text))
#
#     # attar_print(data_length=len(text))
#     chars = sorted(list(set(text)))
#
#     fprint(f'len Chars : {len(chars)}\n', end='\n')
#     s_to_i = {ch: i for i, ch in enumerate(chars)}
#     fprint('Created String to integer Vocab ~ Successfully !!\n')
#     i_to_s = {i: ch for i, ch in enumerate(chars)}
#     fprint('Created integer to String Vocab ~ Successfully  !!\n')
#
#     # encode = lambda s: [i_to_s[c] for c in s]
#     encode = lambda s: [s_to_i[c] for c in s]
#
#     # decode = lambda l: ''.join([s_to_i[c] for c in l])
#     decode = lambda l: ''.join([i_to_s[i] for i in l])
#     text = torch.tensor(encode(text), dtype=torch.long)
#
#     train_data = text[:split]
#     eval_data = text[split:]
#     ptt_text = 'Wellcome To PTT or Poetry Trained Transformer'
#     if pre_show_chunk:
#         fprint(f'Example for word [{ptt_text}]')
#         fprint(encode(ptt_text))
#         fprint(decode(encode(ptt_text)))
#     modes = ['train', "eval"]
#     if pre_show_chunk:
#         train_chunk_x = text[:chunk_size]
#         train_chunk_y = text[1:chunk_size + 1]
#
#         for t in range(chunk_size):
#             x = train_chunk_x[:t + 1]
#             y = train_chunk_y[t]
#
#             fprint(
#                 f'Decoded \nInput : {x} | target : {y}\nEncoded\nInput : {encode(x)} | target : {encode(y)}\n---------------')
#     else:
#         fprint(f'[SKIP] PreShow Status is OFF ! ')
#
#     get_batch = GB(train_data=train_data, eval_data=eval_data, batch_size=batch_size, chunk_size=chunk_size)
#
#     # xb, yb = get_batch('train')
#     # for b in range(batch_size):
#     #     for t in range(chunk_size):
#     #         context = xb[b, :t + 1]
#     #         target = yb[b, t]
#     #         print(f"when input is {context.tolist()} the target: {target}")
#
#     m = PTTMultiHeadAttention(vocab_size=len(chars), chunk_size=chunk_size, number_of_embedded=number_of_embedded,
#                               head_size=head_size,
#                               number_of_layers=number_of_layers,
#                               number_of_head=number_of_head)
#
#     m = m.to(device)
#     optimizer = torch.optim.AdamW(m.parameters(), lr)
#
#     fprint(f'[[ Model Created with {sum(p.numel() for p in m.parameters()) / 1e6} M parameters Over All ]]',
#            color='\033[1;32m')
#     # v = m.generate(torch.zeros((1, 1), dtype=torch.long), 100)
#     # fprint(decode(v[0].tolist()))
#
#     if load_weights:
#         fprint(
#             'Loading Model From Previous Checkpoint ...'
#         )
#         if not os.path.exists(path_weights):
#             fprint(f"Wrong Path To Load Weight This file Doesn't Exist :: => {path_weights}")
#             KeyboardInterrupt()
#         m_s = torch.load(path_weights, device)
#         m.load_state_dict(m_s['model'])
#         optimizer.load_state_dict(m_s['optim'])
#         fprint('Loaded Successfully .')
#     if use_compile:
#         fprint('Compiling Model For Speed Boost 🚀 ...')
#         m = torch.compile(m)
#         fprint('Model Compiled Successfully 🧠')
#     last_eval_loss = 'NONE'
#     for epoch in range(epochs):
#         for mode in modes:
#             x, y = get_batch('train')
#             x, y = x.to(device), y.to(device)
#             # print(f'xShape : {x.shape}')
#
#             if mode not in ['eval', 'test']:
#                 m.train()
#                 predict, loss = m(x, y)
#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 optimizer.step()
#             else:
#                 m.eval()
#                 predict, loss = m(x, y)
#                 last_eval_loss = loss.item()
#             fprint(
#                 f'\rEpoch [{epoch + 1}/{epochs}] | Loss : [{loss.item()}] | Mode : [{mode}] | Last Evaluation Loss : [{last_eval_loss}]',
#                 end='')
#             if (epoch + 1) % 500 == 0:
#                 if mode == 'train':
#                     print()
#                     save_model('ptt-m.pt', model=m.state_dict(), epochs=epochs, epoch=epoch + 1, lr=lr,
#                                optim=optimizer.state_dict())
#                 # saves = {
#                 #     'model': m.state_dict(),
#                 #     'epochs': epochs,
#                 #     'epoch': epoch + 1,
#                 #     'lr': lr,
#                 #     'optim': optimizer.state_dict()
#                 # }
#
#                 # torch.save(saves, 'model.pt')
#
#             if (epoch + 1) % 1000 == 0:
#                 if mode == 'eval':
#                     fprint(f'Generating Some Samples To generated-{epoch + 1}.txt')
#                     stream = open(f'generated-{epoch + 1}.txt', 'w')
#                     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#                     stream.write(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
#
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     stream = open('generated.txt', 'w')
#     txt = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
#     stream.write(txt)
