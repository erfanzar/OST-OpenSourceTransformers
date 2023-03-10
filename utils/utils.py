import os
import typing

import torch
import tqdm
from erutils import fprint
from torch.utils.data import Dataset
import time
from tqdm.auto import tqdm
import psutil
from transformers import BertTokenizer, GPT2Tokenizer
from modules.modeling_llmpu import LLmPUConfig
from modules.cross_modules import LLmPConfig
from modules.modelling_llama import LLamaConfig


class Tokens:
    eos = '<|endoftext|>'
    pad = '<|pad|>'
    sos = '<|startoftext|>'
    atn_start = '<|STN|>'
    atn_end = '<|ETN|>'


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


def count_model_parameters(model, div: int = 1e6):
    return sum(m.numel() for m in model.parameters()) / div


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


class DatasetPGTC(Dataset, Tokens):
    def __init__(self, data=None,
                 mode: str = "gpt2", chunk: int = 184,
                 ):
        super().__init__()

        self.input_ids = []
        self.attention_mask = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(mode, bos_token=self.sos, eos_token=self.eos,
                                                       pad_token=self.pad)
        self.chunk = chunk
        self.vocab_size = self.tokenizer.vocab_size
        self.data = data
        if self.data is not None:
            for d in tqdm(self.data):
                if d != '' and not d.startswith(' ='):
                    emb = self.tokenizer.encode_plus(self.sos + d + self.eos, truncation=True, return_tensors='pt',
                                                     max_length=chunk, padding="max_length")
                    self.attention_mask.append(emb['attention_mask'])
                    self.input_ids.append(emb['input_ids'])

    def __len__(self):
        return len(self.input_ids)

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

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item]

    def decode(self, text):
        text = self.tokenizer.decode(text[0], skip_special_tokens=False)
        return text


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


class GPT2Dataset(Dataset, Tokens):

    def __init__(self, txt_list: typing.Optional[typing.List[str]], tokenizer, max_length: typing.Optional[int] = 768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer(self.sos + txt + self.eos, truncation=True,
                                       max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

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


class PGTConfig:
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)


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
        self.attn_dropout: typing.Optional[float] = kwargs.pop('attn_dropout', 0.13)
        self.residual_dropout: typing.Optional[float] = kwargs.pop('residual_dropout', 0.18)
        self.activation: typing.Optional[str] = kwargs.pop('activation', "gelu_new")
        self.embedded_dropout: typing.Optional[float] = kwargs.pop('embedded_dropout', 0.1)
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


def make2d(tensor) -> typing.Optional[torch.Tensor]:
    return tensor.view(-1, tensor.size(-1))


def get_config_by_name(name: str, vocab_size: int = 5000,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> typing.Union[
    HyperParameters, LLamaConfig, LLmPConfig, LLmPUConfig]:
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
                   'PGT-J-large', 'PGT-J-X', 'LLama', 'LLmP', 'LLmP-small', 'LLmPU-small']
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
            num_embedding=720,
            num_heads=12,
            epochs=1000,
            num_layers=12,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            lr=3e-4,
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
            lr=3e-4,
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
            lr=3e-4,
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
    elif name == 'LLama':
        return LLamaConfig(
            vocab_size=vocab_size,
            max_batch_size=3,
            n_layers=8,
            n_heads=8,
            hidden_size=768,
            max_sentence_length=256
        )
    elif name == 'LLmP-small':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=256,
            max_sentence_length=128
        )
    elif name == 'LLmP':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=512,
            max_sentence_length=256
        )
    elif name == 'LLmP-X':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=1280,
            max_sentence_length=256
        )
    elif name == 'LLmP-L':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=1536,
            max_sentence_length=1024
        )
    elif name == 'LLmP-LX':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=18,
            n_heads=16,
            hidden_size=2048,
            max_sentence_length=1024
        )
    elif name == 'LLmPU-base':
        L = {"d_ff": 1024,
             "d_kv": 32,
             "d_model": 512,
             "decoder_start_token_id": 0,
             "dropout_rate": 0.1,
             "eos_token_id": 1,
             "feed_forward_proj": "gated-gelu",
             "initializer_factor": 1.0,
             "is_encoder_decoder": True,
             "layer_norm_epsilon": 1e-06,
             "model_type": "t5",
             "n_positions": 1024,
             "num_decoder_layers": 8,
             "num_heads": 12,
             "num_layers": 8,
             "output_past": True,
             "pad_token_id": 0,
             "relative_attention_max_distance": 128,
             "relative_attention_num_buckets": 32,
             "tie_word_embeddings": False,
             "use_cache": True,
             "max_length": 512,
             "mesh": 5,
             "vocab_size": vocab_size
             }

        return LLmPUConfig(
            **L
        )

    elif name == 'LLmPU-small':
        L = {"d_ff": 1792,
             "d_kv": 32,
             "d_model": 256,
             "decoder_start_token_id": 0,
             "dropout_rate": 0.1,
             "eos_token_id": 1,
             "feed_forward_proj": "gated-gelu",
             "initializer_factor": 1.0,
             "is_encoder_decoder": True,
             "layer_norm_epsilon": 1e-06,
             "model_type": "t5",
             "n_positions": 1024,
             "num_decoder_layers": 6,
             "num_heads": 12,
             "num_layers": 6,
             "output_past": True,
             "pad_token_id": 0,
             "relative_attention_max_distance": 128,
             "relative_attention_num_buckets": 32,
             "tie_word_embeddings": False,
             "use_cache": True,
             "max_length": 256,
             "mesh": 5,
             "vocab_size": vocab_size
             }
        return LLmPUConfig(
            **L
        )
    elif name == 'LLmPU-large':
        L = {"d_ff": 2048,
             "d_kv": 64,
             "d_model": 768,
             "decoder_start_token_id": 0,
             "dropout_rate": 0.1,
             "eos_token_id": 1,
             "feed_forward_proj": "gated-gelu",
             "initializer_factor": 1.0,
             "is_encoder_decoder": True,
             "layer_norm_epsilon": 1e-06,
             "model_type": "t5",
             "n_positions": 1024,
             "num_decoder_layers": 10,
             "num_heads": 12,
             "num_layers": 10,
             "output_past": True,
             "pad_token_id": 0,
             "relative_attention_max_distance": 128,
             "relative_attention_num_buckets": 32,
             "tie_word_embeddings": False,
             "use_cache": True,
             "max_length": 768,
             "mesh": 5,
             "vocab_size": vocab_size
             }
        return LLmPUConfig(
            **L
        )
    elif name == 'LLmPU-largeX':
        L = {"d_ff": 2048,
             "d_kv": 64,
             "d_model": 768,
             "decoder_start_token_id": 0,
             "dropout_rate": 0.1,
             "eos_token_id": 1,
             "feed_forward_proj": "gated-gelu",
             "initializer_factor": 1.0,
             "is_encoder_decoder": True,
             "layer_norm_epsilon": 1e-06,
             "model_type": "t5",
             "n_positions": 1024,
             "num_decoder_layers": 14,
             "num_heads": 12,
             "num_layers": 14,
             "output_past": True,
             "pad_token_id": 0,
             "relative_attention_max_distance": 128,
             "relative_attention_num_buckets": 32,
             "tie_word_embeddings": False,
             "use_cache": True,
             "max_length": 768,
             "mesh": 5,
             "vocab_size": vocab_size
             }
        return LLmPUConfig(
            **L
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
    memory = psutil.virtual_memory()
    free, total_gpu = torch.cuda.mem_get_info('cuda:0')
    used_gpu = total_gpu - free
    fprint(
        f'DEVICES : [ {torch.cuda.get_device_name()} ] | [ Free : {free / 1e9} GB ] | [ Used : {used_gpu / 1e9} GB ] | '
        f'[ Total : {total_gpu / 1e9} GB ]\n'
        f'RAM : [ Free : {memory.free / 1e9} GB ] | [ Total : {memory.total / 1e9} GB ]')


def get_memory(index: int) -> typing.Tuple[float, float, float]:
    """
    :param index: cuda index
    :return: free,used_gpu,total_gpu memory
    """
    free, total_gpu = torch.cuda.mem_get_info(f'cuda:{index}')
    used_gpu = total_gpu - free
    free, total_gpu, used_gpu = free / 1e9, total_gpu / 1e9, used_gpu / 1e9
    return free, used_gpu, total_gpu


def monitor_function(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        print(f'\033[1;92m {function.__name__} took {end - start:.6f} seconds to complete')
        return result

    return wrapper
