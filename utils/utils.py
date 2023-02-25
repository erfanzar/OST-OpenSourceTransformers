import os
import typing

import torch
import tqdm
from erutils import fprint
from torch.utils.data import Dataset
from transformers import BertTokenizer, GPT2Tokenizer


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


def save_checkpoints(name: str = 'model_save.pt', **kwargs):
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
                 mode: str = "gpt2", chunk: int = 184, call_init: bool = True,
                 pt_data: bool = True):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(mode)
        self.chunk = chunk + 2
        self.vocab_size = self.tokenizer.vocab_size
        self.src = src
        self.batch_size = batch_size
        self.pt_data = pt_data
        self.data = None
        if call_init:
            if pt_data:
                self.init_pt(self.src)
            else:
                self.init()

    def __len__(self):
        return ((len(self.src) // self.chunk) - (
                self.batch_size * 2) if self.src is not None else 1) if not self.pt_data else self.data.shape[0]

    @staticmethod
    def load_pt(path: [str, os.PathLike]):
        data = torch.load(path)
        return data

    def init_pt(self, path: typing.Union[str, os.PathLike]):
        if isinstance(path, str):
            path = [path]
        data = torch.cat([torch.load(p) for p in path], dim=0)
        self.data = data

    def encode(self, text):
        enc_trg = self.tokenizer.encode_plus(
            text=text,
            max_length=self.chunk,
            padding='do_not_pad',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return enc_trg

    def init(self):
        if not self.pt_data:
            start_from: int = 0
            data_list = torch.tensor([])
            total = (len(self.src) // self.chunk) - (self.batch_size * 2)
            loop = tqdm.tqdm(iterable=range(start_from, total))

            for ipa in loop:
                data = self.tokenizer.encode_plus(
                    text=self.src[self.chunk * (ipa + 1):],
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    padding='do_not_pad',
                    # max_length=self.chunk,
                    truncation=False
                )['input_ids']
                print(data.shape)
                data_list = torch.cat([data_list, torch.cat([data[:, 0:-2], data[:, 1:-1]], dim=-2).unsqueeze(0)],
                                      dim=-3)
                # print(f'\r\033[1;32m Loading Data [{ipa}/{total}]', end='')

            self.data = data_list
        else:
            raise ValueError('You can\'t use init model when your data type is pt')

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


class HyperParameters(object):
    def __init__(self, **kwargs):
        self.model_type: typing.Optional[str] = kwargs.pop('model_type', 'PGT-s')
        self.num_embedding: typing.Optional[int] = kwargs.pop('num_embedding', 512)
        self.intermediate_size: typing.Optional[int] = kwargs.pop('intermediate_size', 4)
        self.num_heads: typing.Optional[int] = kwargs.pop('num_heads', 8)
        self.chunk: typing.Optional[int] = kwargs.pop('chunk', 256)
        self.vocab_size: typing.Optional[int] = kwargs.pop('vocab_size', 5000)
        self.num_layers: typing.Optional[int] = kwargs.pop('num_layers', 2)
        self.scale_attn_by_layer_idx: typing.Optional[bool] = kwargs.pop('scale_attn_by_layer_idx', False)
        self.use_mask: typing.Optional[bool] = kwargs.pop('use_mask', True)
        self.attn_dropout: typing.Optional[float] = kwargs.pop('attn_dropout', 0.1)
        self.residual_dropout: typing.Optional[float] = kwargs.pop('residual_dropout', 0.2)
        self.activation: typing.Optional[str] = kwargs.pop('activation', "gelu_new")
        self.embedded_dropout: typing.Optional[float] = kwargs.pop('embedded_dropout', 0.15)
        self.epochs: typing.Optional[int] = kwargs.pop('epochs', 500)
        self.lr: typing.Optional[float] = kwargs.pop('lr', 4e-4)
        self.pad_token_id: typing.Optional[int] = kwargs.pop('pad_token_id', 0)
        self.create_attention_mask: typing.Optional[bool] = kwargs.pop('create_attention_mask', False)
        self.device: typing.Optional[str] = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay: typing.Optional[float] = kwargs.pop('weight_decay', 2e-1, )
        for k, v in kwargs.items():
            if k not in self:
                setattr(self, k, v)


"""
dont use this function anymore use hyper parameters instance of this
"""


# def create_config(
#         model_type: str = 'PGT-s',
#         num_embedding: int = 512,
#         num_heads: int = 8,
#         chunk: int = 256,
#         vocab_size: int = 5000,
#         num_layers: int = 2,
#         scale_attn_by_layer_idx: bool = False,
#         use_mask: bool = True,
#         attn_dropout: float = 0.1,
#         residual_dropout: float = 0.2,
#         activation: str = "gelu_new",
#         embd_pdrop: float = 0.15,
#         epochs: int = 500,
#         lr: float = 4e-4,
#         pad_token_id: int = 0,
#         create_attention_mask: bool = False,
#         device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
#         weight_decay: float = 2e-1,
#         **kwargs
#
# ):
#     intermediate_size: int = num_embedding * 4
#     hidden_size: int = num_embedding
#     max_len = chunk
#     max_position_embeddings = max_len
#     ttl = ['max_position_embeddings', 'hidden_size',
#            'intermediate_size', 'device', 'lr', 'chunk',
#            'embd_pdrop', 'activation', 'epochs', 'pad_token_id',
#            'create_attention_mask',
#            'residual_dropout', 'attn_dropout', 'weight_decay',
#            'use_mask', 'scale_attn_by_layer_idx',
#            'num_layers', 'vocab_size',
#            'max_len', 'num_heads', 'num_embedding']
#     cash = CF()
#     for t in ttl:
#         cash.__setattr__(t, eval(t))
#     v = {**kwargs}
#     if len(v) != 0:
#         for k, v in v.items():
#             cash.__setattr__(k, v)
#
#     return cash


def make2d(tensor):
    return tensor.view(-1, tensor.size(-1))


def get_config_by_name(name: str, vocab_size: int = 5000,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> HyperParameters:
    """
    :param device: device for model
    :param vocab_size: vocab_size
    :param name: name of the type of model you want to get config
    [chooses] = ['PGT-ss']['PGT-s']['PGT-m']['PGT-x']['PGT-l']['PGT-A']
    :return: Config
    """

    """
        self.num_embedding: int = kwargs.pop('num_embedding', 512)
        self.num_heads: int = kwargs.pop('num_heads', 8)
        self.chunk: int = kwargs.pop('chunk', 256)
        self.vocab_size: int = kwargs.pop('vocab_size', 5000)
        self.num_layers: int = kwargs.pop('num_layers', 2)
        self.scale_attn_by_layer_idx: bool = kwargs.pop('scale_attn_by_layer_idx', False)
        self.use_mask: bool = kwargs.pop('use_mask', True)
        self.attn_dropout: float = kwargs.pop('attn_dropout', 0.1)
        self.residual_dropout: float = kwargs.pop('residual_dropout', 0.2)
        self.activation: str = kwargs.pop('activation', "gelu_new")
        self.embedded_dropout: float = kwargs.pop('embedded_dropout', 0.15)
        self.epochs: int = kwargs.pop('epochs', 500)
        self.lr: float = kwargs.pop('lr', 4e-4)
        self.pad_token_id: int = kwargs.pop('pad_token_id', 0)
        self.create_attention_mask: bool = kwargs.pop('create_attention_mask', False)
        self.device: str = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay: float = kwargs.pop('weight_decay', 2e-1, )
    """
    models_name = ['PGT-Cs', 'PGT-As', 'PGT-s', 'PGT-m', 'PGT-x', 'PGT-l', 'PGT-A', 'PGT-J-small', 'PGT-J-medium',
                   'PGT-J-large', 'PGT-J-X']
    if name == 'PGT-Cs':
        return HyperParameters(
            model_type=name,
            num_embedding=360,
            num_heads=10,
            epochs=1000,
            num_layers=8,
            device=device,
            vocab_size=vocab_size,
            chunk=184,
            lr=3e-4,
            use_mask=True
        )
    if name == 'PGT-As':
        return HyperParameters(
            model_type=name,
            num_embedding=624,
            num_heads=12,
            epochs=1000,
            num_layers=10,
            device=device,
            vocab_size=vocab_size,
            chunk=184,
            lr=4e-4,
            use_mask=True
        )
    elif name == 'PGT-s':
        return HyperParameters(
            model_type=name,
            num_embedding=256,
            num_heads=8,
            num_layers=4,
            device=device,
            vocab_size=vocab_size,
            chunk=64,
            use_mask=True
        )
    elif name == 'PGT-m':
        return HyperParameters(
            model_type=name,
            num_embedding=512,
            num_heads=8,
            num_layers=8,
            device=device,
            vocab_size=vocab_size,
            chunk=184,
            use_mask=True
        )
    elif name == 'PGT-x':
        return HyperParameters(
            model_type=name,
            num_embedding=512,
            num_heads=16,
            num_layers=14,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-l':

        return HyperParameters(
            model_type=name,
            num_embedding=728,
            num_heads=14,
            num_layers=20,
            epochs=1000,
            device=device,
            vocab_size=vocab_size,
            chunk=184,
            lr=4e-4,
            use_mask=True
        )
    elif name == 'PGT-A':
        prp = torch.cuda.get_device_properties("cuda")
        print(f'\033[1;32mWarning You Loading the Largest Model on {prp.name} : {prp.total_memory / 1e9} GB')
        return HyperParameters(
            model_type=name,
            num_embedding=1024,
            num_heads=32,
            num_layers=48,
            epochs=1000,
            device=device,
            vocab_size=vocab_size,
            chunk=184,
            lr=4e-4,
            use_mask=True
        )
    elif name == 'PGT-J-small':
        return HyperParameters(
            model_type=name,
            num_embedding=512,
            num_heads=16,
            num_layers=10,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-J-medium':
        return HyperParameters(
            model_type=name,
            num_embedding=512,
            num_heads=16,
            num_layers=18,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-J-large':
        return HyperParameters(
            model_type=name,
            num_embedding=984,
            num_heads=24,
            num_layers=16,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-J-X':
        return HyperParameters(
            model_type=name,
            num_embedding=1712,
            num_heads=107,
            num_layers=38,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    else:
        raise NameError(
            f"Valid Names for Model are {models_name} | [ERROR : Unknown {name} type]")


def print_config(config: typing.Optional[HyperParameters]) -> None:
    fprint('Loaded Configs :: =>')
    for d in config.__dict__:
        print('{:<25} : {:>25}'.format(f"{d}", f"{config.__dict__[d]}"))


def device_info() -> None:
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
