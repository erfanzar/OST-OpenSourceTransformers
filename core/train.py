import os
import typing

import torch
from erutils.loggers import fprint
from erutils.utils import read_yaml, read_json
from torch.utils.data import DataLoader

from modules.models import PTTGenerative
from utils.utils import DatasetQA, save_model


# from torch.utils.tensorboard import SummaryWriter

def train_ptt_g(config_path: typing.Union[str, os.PathLike],
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    ...


def train_ptt(config_path: typing.Union[str, os.PathLike],
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    cfg = read_yaml(config_path)
    data_path = cfg['data_path']
    epochs = cfg['epochs']
    lr = float(cfg['lr'])
    max_length = cfg['chunk']
    number_of_heads = cfg['number_of_heads']
    number_of_layers = cfg['number_of_layers']
    embedded = cfg['embedded']
    use_train = cfg['train']
    batch_size = cfg['batch_size']
    # ssm = SummaryWriter(log_dir='results/out')
    data = read_json(data_path)

    questions = [data[v]['question'] for v in data]
    answers = [data[v]['answer'] for v in data]

    dataset = DatasetQA(max_length=max_length, src=questions, trg=answers)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    vocab_size: int = dataset.vocab_size
    pad_index: int = dataset.pad_token_id
    eos: int = dataset.tokenizer.eos_token_id
    ptt = PTTGenerative(
        vocab_size=vocab_size,
        chunk=max_length,
        embedded=embedded,
        eos=eos,
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        pad_index=pad_index,
    ).to(device)
    fprint(f'[[ Model Created with {sum(p.numel() for p in ptt.parameters()) / 1e6} M parameters Over All ]]',
           color='\033[1;32m')
    optimizer = torch.optim.AdamW(ptt.parameters(), lr)


    data_ip = dataset.__getitem__(1)
    answer = data_ip[1].to(device)
    question = data_ip[0].to(device)
    print(f'QUESTION : {dataset.decode(question)}')
    print(f'ANSWER   : {dataset.decode(answer)}')
    sos = dataset.sos().to(device)
    # print(f'AW : {answer.shape}')
    for epoch in range(epochs):
        lsa = 0
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device).view(-1, src.size(-1))
            trg = trg.to(device)
            # print(trg.shape)
            _, losses = ptt.forward(src=src, trg=sos.repeat(batch_size, 1), target=trg[:, :, :-1])
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lsa += losses.item()
            fprint(
                f'\rEPOCH : [{epoch}/{epochs}] | LOSS : {losses.item() / batch_size} | EPOCH LOSS AVG : {(lsa / (i + 1)) / batch_size} | ITER : {i + 1}',
                end='')
            if i == 500:
                break
        print()
        if epoch % 5 == 0:
            print()
            predictions = ptt.generate(src=question, idx=sos)
            fprint(f'QUESTION : {dataset.decode(question)}')
            fprint(f'ANSWER   : {dataset.decode(answer)}')
            fprint(f'PREDICTION : {dataset.decode(predictions)}')
            save_model(model=ptt.state_dict(), optimizer=optimizer.state_dict(), epochs=epochs, epoch=epoch,
                       name='model.pt')
            fprint('==> MODEL SAVE SUCCESSFULLY')
    print()
    predictions = ptt.generate(src=question, idx=sos)
    print(f'PREDICTION : {dataset.decode(predictions)}')