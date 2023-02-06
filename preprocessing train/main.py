import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from models import PTT

ssm = SummaryWriter(log_dir='./out')


class DatasetQA(Dataset):
    def __init__(self, src=None, trg=None, mode: str = "bert-base-uncased", max_length: int = 512,
                 pad_to_max_length: bool = True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(mode)

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_to_max_length = pad_to_max_length
        self.src = src
        self.max_length = max_length
        self.trg = trg

    def __len__(self):
        return len(self.src) if self.src is not None else 1

    def __getitem__(self, item):
        src = str(self.src[item])
        trg = str(self.trg[item]['text'][0])
        enc_src = self.tokenizer.encode_plus(
            text=src,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # padding='longest',
            # return_length=True,
            padding='longest' if not self.pad_to_max_length else 'max_length',
            truncation=True

        )
        enc_trg = self.tokenizer.encode_plus(
            text=trg,
            max_length=self.max_length + 1,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            # return_length=True,
            padding='longest' if not self.pad_to_max_length else 'max_length',

            truncation=True

        )
        return [enc_src['input_ids'], enc_trg['input_ids']]

    def decode(self, text):
        text = self.tokenizer.decode(text[0], skip_special_tokens=False)
        return text


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)


max_length: int = 256
embedded: int = 256
number_of_heads: int = 4
number_of_layers: int = 6
# dataset = DatasetQA(max_length=max_length)

if __name__ == "__main__":
    squad_dataset = load_dataset('squad')
    train_data = squad_dataset['train']
    data_len = train_data.num_rows
    questions = train_data.data['question']
    answers = train_data.data['answers']
    dataset = DatasetQA(max_length=max_length, src=questions[:800], trg=answers[:800])
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
            print(f'\033[1;36m\r[{epoch}/{epochs}] | Loss : {loss.item()} | Iter : {i}', end='')
            if i % 20 == 0:
                # example_question = dataset.decode(x[0])
                # example_answer = dataset.decode(y[0])
                example_question = dataset.decode(x)
                example_answer = dataset.decode(y)
                # print(predict.shape)
                predict_sa = torch.multinomial(torch.softmax(predict[0], dim=-1), num_samples=1).view(1, -1)
                prra = dataset.decode(predict_sa)
                ssm.add_text('QUESTION', example_question, i)
                ssm.add_text('ANSWER', example_answer, i)

                ssm.add_text('PREDICT', prra, i)
                ssm.add_scalar('train/LOSS', loss.item(), i)
            # dataset.brea()

        print('\n')

        save_model(model=ptt.state_dict(), optimizer=optimizer.state_dict(), epochs=epochs, epoch=epoch,
                   name='TDFA.pt')
