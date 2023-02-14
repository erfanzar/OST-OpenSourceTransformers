import dataclasses

import torch
import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer


class GB:
    def __init__(self, train_data, eval_data, batch_size, chunk_size):
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def forward(self, mode: str, *args, **kwargs):
        data = self.train_data if mode == 'train' else self.eval_data
        ix = torch.randint(len(data) - self.chunk_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.chunk_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.chunk_size + 1] for i in ix])
        return x, y


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)


def tokenize_words(word: list, first_word_token: int = 0, swap: int = 1001, last_word_token: int = 1002,
                   pad_index: int = 1003):
    """
    :param swap:
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: 0 for start token | 1002 for end token
    """
    word = [(swap if w == 0 else w) for w in word]
    word = [first_word_token] + word
    word.append(last_word_token)
    word.append(pad_index)
    return word


def detokenize_words(word: list, first_word_token: int = 0, last_word_token: int = 1002, pad_index: int = 1003):
    """
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: un tokenized words
    """

    w = [(first_word_token if w == last_word_token - 1 else w) for w in
         [w for w in word if w not in [last_word_token, first_word_token]]]
    del w[-1]
    # print(f'W : {w}')
    return w


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


def create_config(
        model_type: str = 'PGT-s',
        num_embedding: int = 512,
        num_heads: int = 8,
        chunk: int = 256,
        vocab_size: int = 5000,
        num_layers: int = 2,
        scale_attn_by_layer_idx: bool = False,
        use_mask: bool = True,
        attn_dropout: float = 0.2,
        residual_dropout: float = 0.2,
        activation: str = "gelu_new",
        embd_pdrop: float = 0.1,
        epochs: int = 500,
        lr: float = 4e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        weight_decay: float = 2e-1,
        **kwargs

):
    intermediate_size: int = num_embedding * 4
    hidden_size: int = num_embedding
    max_len = chunk
    max_position_embeddings = max_len
    ttl = ['max_position_embeddings', 'hidden_size',
           'intermediate_size', 'device', 'lr', 'chunk',
           'embd_pdrop', 'activation', 'epochs',
           'residual_dropout', 'attn_dropout', 'weight_decay',
           'use_mask', 'scale_attn_by_layer_idx',
           'num_layers', 'vocab_size',
           'max_len', 'num_heads', 'num_embedding']
    cash = CF()
    for t in ttl:
        cash.__setattr__(t, eval(t))
    v = {**kwargs}
    if len(v) != 0:
        for k, v in v.items():
            cash.__setattr__(k, v)

    return cash


def make2d(tensor):
    return tensor.view(-1, tensor.size(-1))


def get_config_by_name(name: str = 'PGT-s', vocab_size: int = 5000,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> create_config:
    """
    :param device: device for model
    :param vocab_size: vocab_size
    :param name: name of the type of model you want to get config
    [chooses] = ['PGT-ss']['PGT-s']['PGT-m']['PGT-x']['PGT-l']['PGT-A']
    :return: Config
    """

    if name == 'PGT-As':
        return create_config(
            name,
            num_embedding=624,
            num_heads=12,
            epochs=1000,
            num_layers=10,
            device=device,
            vocab_size=vocab_size,
            chunk=128,
            lr=4e-4,
            use_mask=True
        )
    elif name == 'PGT-s':
        return create_config(
            name,
            num_embedding=256,
            num_heads=8,
            num_layers=4,
            device=device,
            vocab_size=vocab_size,
            chunk=64,
            use_mask=True
        )
    elif name == 'PGT-m':
        return create_config(
            name,
            num_embedding=512,
            num_heads=8,
            num_layers=8,
            device=device,
            vocab_size=vocab_size,
            chunk=128,
            use_mask=True
        )
    elif name == 'PGT-x':
        return create_config(
            name,
            num_embedding=512,
            num_heads=16,
            num_layers=14,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-l':
        return create_config(
            name,
            num_embedding=728,
            num_heads=14,
            num_layers=20,
            device=device,
            vocab_size=vocab_size,
            chunk=512,
            use_mask=True
        )
    elif name == 'PGT-A':
        prp = torch.cuda.get_device_properties("cuda")
        print(f'\033[1;32mWarning You Loading the Largest Model on {prp.name} : {prp.total_memory / 1e9} GB')
        return create_config(
            name,
            num_embedding=1024,
            num_heads=32,
            num_layers=42,
            device=device,
            vocab_size=vocab_size,
            chunk=728,
            use_mask=True
        )
    else:
        raise NameError(
            f"Valid Names for Model are ['PGT-s']['PGT-m']['PGT-x']['PGT-l']['PGT-A'] | [ERROR : Unknown {name} type]")
