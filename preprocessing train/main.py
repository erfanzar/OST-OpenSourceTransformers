from models import PTT
from datasets import load_dataset
import torch
import torch.nn as nn
from erutils.nlp import Lang
from cms import add_pad, add_special_tokens
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import BertTokenizer


class DatasetQA(Dataset):
    def __init__(self, src, trg, mode: str = "bert-base-uncased", max_length: int = 512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(mode)

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src = src
        self.max_length = max_length
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = str(self.src[item])
        trg = str(self.trg[item]['text'][0])
        enc_src = self.tokenizer.encode_plus(
            text=src,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # return_length=True,
            pad_to_max_length=True,
            truncation=True

        )
        enc_trg = self.tokenizer.encode_plus(
            text=trg,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # return_length=True,
            pad_to_max_length=True,
            truncation=True

        )
        return enc_src['input_ids'], enc_trg['input_ids']


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)


if __name__ == "__main__":
    squad_dataset = load_dataset('squad')
    train_data = squad_dataset['train']
    data_len = train_data.num_rows
    questions = train_data.data['question']
    answers = train_data.data['answers']
    max_length: int = 256
    embedded: int = 256
    number_of_heads: int = 2
    number_of_layers: int = 4
    dataset = DatasetQA(max_length=max_length, src=questions, trg=answers)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)
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
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = x.to(device)
            # print(f'X SHAPE : {x.shape} "|" Y SHAPE : {y.shape}')
            trg = y[:, :, :]
            #
            ys = y[:, :, :].contiguous().view(-1)

            # print(f'TARGET : {trg} "|" Y : {ys}')

            predict = ptt(x, trg)
            optimizer.zero_grad()
            loss = F.cross_entropy(predict.view(-1, predict.size(-1)), ys, ignore_index=pad_index)
            loss.backward()
            optimizer.step()
            print(f'\r[{epoch}/{epochs}] | Loss : {loss.item()} | Iter : {i}', end='')
        print('\n')

        save_model(model=ptt.state_dict(), optimizer=optimizer.state_dict(), epochs=epochs, epoch=epoch,
                   name='TDFA.pt')
