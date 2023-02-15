from torch.utils.data import Dataset
from transformers import BertTokenizer


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

    def encode(self, text, padding: bool = False):
        enc_trg = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='longest' if padding else padding,
            truncation=True
        )
        return enc_trg

    def __getitem__(self, item):
        # src = str(self.src[item])
        # trg = str(self.trg[item]['text'][0])
        src = str(self.src[item])
        trg = str(self.trg[item])
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
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,

            return_tensors='pt',

            padding='longest' if not self.pad_to_max_length else 'max_length',

            truncation=True

        )
        # it = {
        #     'input': enc_src['input_ids'],
        #     'label': enc_trg['input_ids']
        # }
        it = {
            'input': enc_src,
            'label': enc_trg
        }
        return it

    def decode(self, text):
        text = self.tokenizer.decode(text[0], skip_special_tokens=True)
        return text

    def sos(self):
        return self.tokenizer.encode_plus(
            text='[CLS]',
            max_length=self.max_length,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt',
            padding='longest' if not self.pad_to_max_length else 'max_length',
            truncation=True
        )['input_ids']


class DatasetPGT(Dataset):
    def __init__(self, src=None, batch_size: int = 4,
                 mode: str = "bert-base-uncased", chunk: int = 128, call_init: bool = False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(mode)
        self.chunk = chunk + 2
        self.vocab_size = self.tokenizer.vocab_size
        self.src = src
        self.batch_size = batch_size
        self.data = None
        if call_init:
            self.init()

    def __len__(self):
        return (len(self.src) // self.chunk) - (self.batch_size * 2) if self.src is not None else 1

    def encode(self, text):
        enc_trg = self.tokenizer.encode_plus(
            text=text,
            max_length=self.chunk,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return enc_trg

    def init(self):
        start_from: int = 0
        data_list = torch.tensor([])
        total = (len(self.src) // self.chunk) - (self.batch_size * 2)
        loop = tqdm.tqdm(iterable=range(start_from, total))

        for ipa in loop:
            data = self.tokenizer.encode_plus(
                text=self.src[self.chunk * (ipa + 1):],
                add_special_tokens=False,
                return_attention_mask=True,
                return_tensors='pt',
                padding='longest',
                max_length=self.chunk,
                truncation=True
            )['input_ids']

            data_list = torch.cat([data_list, torch.cat([data[:, 0:-2], data[:, 1:-1]], dim=-2).unsqueeze(0)], dim=-3)
            # print(f'\r\033[1;32m Loading Data [{ipa}/{total}]', end='')

        self.data = data_list

    def __getitem__(self, item):
        x, y = self.data[item]
        return x.unsqueeze(0), y.unsqueeze(0)

    def decode(self, text):
        text = self.tokenizer.decode(text[0], skip_special_tokens=True)
        return text

    def sos(self):
        return self.tokenizer.encode_plus(
            text='[CLS]',
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )['input_ids']


@dataclasses.dataclass
class CF:
    ...
