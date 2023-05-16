import json
import os
import time
import typing
from importlib.util import find_spec
from typing import Union, Optional

import accelerate
import psutil
import torch
import tqdm
from datasets import load_dataset
from erutils import fprint
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizer, GPT2Tokenizer

from modules import LGeMConfig, LLamaConfig, PGTConfig, LLmPConfig, LLMoUConfig, LLmPUConfig


def available(name: str):
    p_ = find_spec(name)
    if p_ is not None:
        return True
    else:
        return False


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


def save_checkpoints(name: str, **kwargs):
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


def count_model_parameters(model, div: float = 1e6):
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
    HyperParameters, LLamaConfig, LLmPConfig, LLmPUConfig, LLMoUConfig, PGTConfig, LGeMConfig]:
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
    models_name = ['PGT-S',
                   'PGT-M',
                   'PGT-X',
                   'PGT-LX',
                   'PGT-LXX',
                   'LLama',
                   'LLmP-S',
                   'LLmP-ML',
                   'LLmP',
                   'LLmP-X',
                   'LLmP-L',
                   'LLmP-LX',
                   'LLMoU-S',
                   'LLMoU-ML',
                   'LLMoU',
                   'LLMoU-X',
                   'LLMoU-L',
                   'LLMoU-LX'
                   'LLmPU-base',
                   'LLmPU-S',
                   'LLmPU-L',
                   'LLmPU-LX', ]

    if name == 'PGT-S':
        return PGTConfig(
            n_layers=10,
            n_heads=12,
            epochs=500,
            hidden_size=768,
            max_sequence_length=256,
            vocab_size=vocab_size
        )
    elif name == 'PGT-M':
        return PGTConfig(
            n_layers=18,
            n_heads=12,
            epochs=500,
            hidden_size=1024,
            max_sequence_length=512,
            vocab_size=vocab_size
        )
    elif name == 'PGT-X':
        return PGTConfig(
            n_layers=28,
            n_heads=16,
            epochs=500,
            hidden_size=1536,
            max_sequence_length=512,
            vocab_size=vocab_size
        )
    elif name == 'PGT-LX':

        return PGTConfig(
            n_layers=34,
            n_heads=32,
            epochs=500,
            hidden_size=2048,
            max_sequence_length=768,
            vocab_size=vocab_size
        )
    elif name == 'PGT-LXX':
        return PGTConfig(
            n_layers=64,
            n_heads=32,
            epochs=500,
            hidden_size=4096,
            max_sequence_length=2000,
            vocab_size=vocab_size
        )
    elif name == 'LLama':
        return LLamaConfig(
            vocab_size=vocab_size,
            max_batch_size=3,
            n_layers=18,
            n_heads=16,
            hidden_size=4096,
            max_sequence_length=256
        )
    elif name == 'LLmP-S':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=768,

        )
    elif name == 'LLmP-ML':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=18,
            n_heads=16,
            epochs=500,
            hidden_size=1024,

        )
    elif name == 'LLmP':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=24,
            n_heads=16,
            epochs=500,
            hidden_size=1536,

        )
    elif name == 'LLmP-X':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=36,
            n_heads=16,
            epochs=500,
            hidden_size=1792,

        )
    elif name == 'LLmP-L':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=32,
            n_heads=32,
            epochs=500,
            hidden_size=2048,

        )
    elif name == 'LLmP-LX':
        return LLmPConfig(
            vocab_size=vocab_size,
            n_layers=48,
            n_heads=32,
            hidden_size=4096,

        )
    elif name == 'LLMoU-S':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=10,
            n_heads=8,
            epochs=500,
            hidden_size=768,

        )
    elif name == 'LLMoU-ML':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=18,
            n_heads=16,
            epochs=500,
            hidden_size=1024,

        )
    elif name == 'LLMoU':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=26,
            n_heads=16,
            epochs=500,
            hidden_size=1536,
            max_sequence_length=256
        )
    elif name == 'LLMoU-X':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=34,
            n_heads=32,
            epochs=500,
            hidden_size=2048,
            max_sequence_length=256
        )
    elif name == 'LLMoU-L':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=48,
            n_heads=32,
            epochs=500,
            hidden_size=2048,
            max_sequence_length=1024
        )
    elif name == 'LLMoU-LX':
        return LLMoUConfig(
            vocab_size=vocab_size,
            n_layers=52,
            n_heads=32,
            hidden_size=2048,
            max_sequence_length=2048
        )
    elif name == 'LLmPU-base':
        L = {"d_ff": 128 * 20,
             "d_kv": 32 * 3,
             "d_model": 128 * 14,
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

    elif name == 'LLmPU-S':
        L = {"d_ff": 128 * 14,
             "d_kv": 32 * 2,
             "d_model": 128 * 8,
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
    elif name == 'LLmPU-L':
        L = {"d_ff": 128 * 24,
             "d_kv": 32 * 3,
             "d_model": 128 * 14,
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
    elif name == 'LLmPU-LX':
        L = {"d_ff": 6144,
             "d_kv": 128,
             "d_model": 2048,
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
    elif name == 'LGeM-SM':
        return LGeMConfig(
            hidden_size=512,
            intermediate_size=512 * 2,
            num_hidden_layers=6,
            num_attention_heads=8,
            vocab_size=32000,
            max_sequence_length=128
        )
    elif name == 'LGeM-DEBUG':
        return LGeMConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            vocab_size=32000,
        )
    elif name == 'LGeM-S':
        return LGeMConfig(
            hidden_size=768,
            intermediate_size=768 * 7,
            num_hidden_layers=6,
            num_attention_heads=8,
            vocab_size=32000,
        )
    elif name == 'LGeM-ML-OLD':
        return LGeMConfig(
            hidden_size=1344,
            intermediate_size=1344 * 4,
            num_hidden_layers=10,
            num_attention_heads=16,
            vocab_size=-1,
        )
    elif name == 'LGeM-ML':
        return LGeMConfig(
            hidden_size=1408,
            intermediate_size=1408 * 5,
            num_hidden_layers=14,
            num_attention_heads=16,
            vocab_size=-1,
        )
    elif name == 'LGeM':
        return LGeMConfig(
            hidden_size=2048,
            intermediate_size=2048 * 8,
            num_hidden_layers=24,
            num_attention_heads=16,
            vocab_size=32000,
            max_sequence_length=2048
        )
    elif name == 'LGeM-X':
        return LGeMConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            vocab_size=32000,
            max_sequence_length=2048
        )
    elif name == 'LGeM-L':
        return LGeMConfig(
            hidden_size=8192,
            intermediate_size=8192 * 6,
            num_hidden_layers=32,
            num_attention_heads=64,
            vocab_size=32000,
            max_sequence_length=4096
        )
    elif name == 'LGeM-LX':
        return LGeMConfig(
            hidden_size=10240,
            intermediate_size=10240 * 5,
            num_hidden_layers=54,
            num_attention_heads=64,
            vocab_size=32000,
            max_sequence_length=4096
        )
    elif name == 'LGeM-LLX':
        return LGeMConfig(
            hidden_size=12288,
            intermediate_size=12288 * 5,
            num_hidden_layers=92,
            num_attention_heads=128,
            vocab_size=32000,
            max_sequence_length=8192
        )

    else:
        raise NameError(
            f"Valid Names for Model are {models_name} | [ERROR : Unknown {name} type]")


def print_config(config: typing.Optional[HyperParameters]) -> None:
    fprint('Loaded Configs :: =>')
    for d in config.__dict__:
        print('{:<25} : {:>25}'.format(f"{d}", f"{config.__dict__[d]}"))


def device_info() -> None:
    if torch.cuda.is_available():
        prp = torch.cuda.get_device_properties("cuda")
        memory = psutil.virtual_memory()
        free, total_gpu = torch.cuda.mem_get_info('cuda:0')
        used_gpu = total_gpu - free
        fprint(
            f'DEVICES : [ {torch.cuda.get_device_name()} ] | [ Free : {free / 1e9} GB ] | [ Used : {used_gpu / 1e9} GB ] | '
            f'[ Total : {total_gpu / 1e9} GB ]\n'
            f'RAM : [ Free : {memory.free / 1e9} GB ] | [ Total : {memory.total / 1e9} GB ]')
    else:
        memory = psutil.virtual_memory()

        fprint(f'RAM : [ Free : {memory.free / 1e9} GB ] | [ Total : {memory.total / 1e9} GB ]')


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


def create_output_path(path: Union[os.PathLike, str], name: Optional[str]):
    path_exist = os.path.exists(path)
    if not path_exist:
        os.mkdir(path)
    name_exist = os.path.exists(os.path.join(path, name))
    u_name = name
    if not name_exist:
        os.mkdir(os.path.join(path, u_name))

    else:
        at_try: int = 1
        while True:
            try:
                u_name = name + f'_{at_try}'
                os.mkdir(os.path.join(path, u_name))
                break
            except FileExistsError:
                at_try += 1
    return f'{path}/{u_name}'


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.002)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.002)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def get_data(data_src):
    if data_src.endswith('.txt'):
        data = open(data_src, 'r', encoding='utf8').read().split()
    elif data_src.endswith('.json'):
        data = json.load(open(data_src, 'r', encoding='utf8'))
    elif data_src.startswith('HF-'):
        name = data_src.replace('HF-', '')
        if '//' in name:
            model_name = name.split('//')
            data = load_dataset(model_name[0], model_name[1])
        else:
            data = load_dataset(name)

    else:
        data = None
        raise ValueError()
    return data


def compile_model(model: torch.nn.Module):
    try:
        import torch._dynamo as dynamo
        torch._dynamo.config.verbose = True
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
        print("Model compiled set")
    except Exception as err:
        print(f"Model compile not supported: {err}")
    return model


def accelerate_mode(accelerator: accelerate.Accelerator, model: torch.nn.Module = None,
                    optimizer: torch.optim.Optimizer = None, dataloader=None):
    model = accelerator.prepare_model(model) if model is not None else None
    optimizer = accelerator.prepare_optimizer(optimizer) if optimizer is not None else None

    dataloader = accelerator.prepare_data_loader(dataloader) if dataloader is not None else None
    return model, optimizer, dataloader


def prompt_to_instruction(instruction, input_=None, response_=None, eos='<|endoftext|>'):
    if input_ is None:
        st1_prompting = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{instruction}\n\n'
    else:
        st1_prompting = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{instruction}\n\n### Input:\n\n{input_}\n\n'
    resp = f'### Response:\n\n{response_}{eos}' if response_ is not None else '### Response:\n\n'
    return st1_prompting + resp


def generate(model, input_ids_, tokeinzer_, max_length: int = 256, tempeture: float = 1, eos_token_id: int = 2):
    with torch.no_grad():
        before_start = len(input_ids_[0]) + 1
        for _ in range(max_length):
            out = model(
                input_ids=input_ids_,
                return_dict=True,
            )
            opa = torch.nn.functional.softmax(out.logits[:, -1, :] / tempeture)
            namula = torch.multinomial(opa, 1)
            input_ids_ = torch.cat([input_ids_, namula], -1)

            print(f"\r{tokeinzer_.decode(input_ids_[0], skip_special_tokens=True)[before_start:]}", end='')
            if namula[0].item() == eos_token_id:
                break
            yield tokeinzer_.decode(namula[0], skip_special_tokens=True)
    return f"{tokeinzer_.decode(input_ids_[0], skip_special_tokens=True)[before_start:]}"


class Controller:
    def __init__(self, tokenizer, json_url='https://huggingface.co/erfanzar/LGeM-7B-MT/raw/main/non-w-words.json'):
        import json
        import requests
        self.tokenizer = tokenizer
        try:
            response = requests.get(json_url)
            self.total = json.loads(response.text)
        except:
            self.total = None

    def controll_text(self, text):
        if not self.total is None:
            try:
                text = self.tokenizer.encode(text).input_ids
            except AttributeError:
                text = self.tokenizer.encode(text)
            for i, (n, r) in enumerate(zip(self.total['n_words'], self.total['r_words'])):
                nl = self.total['n_words'][n]
                rl = self.total['r_words'][r]
                text = self.replace_pattern(text, nl, rl)
            text = self.tokenizer.decode(text)
            return text
        else:
            return text

    @staticmethod
    def replace_pattern(list_, pattern, replacement):
        new_list = []
        skip_ = 0
        for i, val in enumerate(list_):
            if val == pattern[0]:
                true_false_list = []
                if len(pattern) > 1:
                    for org, pat in zip(list_[i:i + len(pattern)], pattern):
                        true_false_list.append(True if org == pat else False)
                    if set(true_false_list):
                        for r in replacement:
                            new_list.append(r)
                            skip_ += 1
            elif skip_ != 0:
                skip_ -= 1
            else:
                new_list.append(val)
        return new_list


if available('jax') and available('flax'):
    import jax
    import flax
    from jax import numpy as jnp


    def flax_count_params(params):
        _i = flax.core.unfreeze(params)
        _i = jax.tree_util.tree_flatten(_i)[0]
        return sum(i.size for i in _i)


    def cross_entropy_loss(prediction: jnp.DeviceArray, targets: jnp.DeviceArray):
        targets = jax.nn.one_hot(targets, num_classes=prediction.shape[-1])
        prediction = jax.nn.softmax(prediction)
        loss = - jnp.sum(jnp.log(prediction + 1e-12) * targets, axis=-1)
        loss = jnp.mean(loss)
        return loss

from bitsandbytes.nn import Linear8bitLt


def replace_model_with_int8_linear(model: torch.nn.Module, threshold=6.0, module_to_not_convert=""):
    for name, module in model.named_children():
        ln = len(list(module.children()))
        if ln > 0:
            replace_model_with_int8_linear(module, threshold=threshold, module_to_not_convert=module_to_not_convert)
        if isinstance(module, torch.nn.Linear) and name != module_to_not_convert:
            model._modules[name] = Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
    return model


def quantize_tensor(weight, scale=None):
    # Calculate the scaling factor if not provided
    if scale is None:
        max_abs_range = torch.max(torch.abs(weight))
        scale = max_abs_range / 127.0

    # Quantize the tensor to int8
    weight = torch.round(weight.cuda() / scale).clamp(-128, 127).to(torch.int8).cpu()
    return weight


def transform_weight_quantize(state_dict):
    for i, (k, v) in enumerate(state_dict.items()):
        if not k.endswith('bias') and not k.endswith('inv_freq'):
            state_dict[k] = quantize_tensor(v)
            print(f'{i} Quantize {k}')
    return state_dict


def simple_chunk(input_ids_, attention_mask_, chunk=512, drop_last=True):
    input_ids = []
    attention_mask = []

    for current_chunk in range(0, len(attention_mask_), chunk):
        try:
            input_ids.append(input_ids_[current_chunk:current_chunk + chunk])
            attention_mask.append(attention_mask_[current_chunk:current_chunk + chunk])
        except KeyError:
            if not drop_last:
                input_ids.append(input_ids_[current_chunk:])
                attention_mask.append(attention_mask_[current_chunk:])
    if len(input_ids[-1]) != chunk and not drop_last:
        rem = chunk - len(input_ids[-1])
        added_remo = [0 for _ in range(rem)]
        input_ids[-1] += added_remo
        attention_mask[-1] += added_remo
    return input_ids, attention_mask


def set_ff_model(_model, embedding_requires_grad=True):
    mlp, ll, q_p, v_p, k_p, o_p = [], [], [], [], [], []
    last_l = len(_model.model.layers) - 1
    for i, (name, param) in enumerate(_model.named_parameters()):
        if f"{last_l}" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        if 'mlp' in name:
            param.requires_grad = False
        if i == 0 and embedding_requires_grad:
            param.requires_grad = True

    for i, (name, param) in enumerate(_model.named_parameters()):
        print("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
        if 'q_proj' in name:
            q_p.append(param.numel() / 1e6)
        if 'k_proj' in name:
            k_p.append(param.numel() / 1e6)
        if 'v_proj' in name:
            v_p.append(param.numel() / 1e6)
        if 'o_proj' in name:
            o_p.append(param.numel() / 1e6)
        if 'mlp' in name:
            mlp.append(param.numel() / 1e6)
        if f"{last_l}" in name:
            ll.append(param.numel() / 1e6)

    print(f'K Proj Contain {sum(k_p)} Million Parameters')
    print(f'Q Proj Contain {sum(q_p)} Million Parameters')
    print(f'O Proj Contain {sum(o_p)} Million Parameters')
    print(f'V Proj Contain {sum(v_p)} Million Parameters')
    print(f'MLP Contain    {sum(mlp)} Million Parameters')

    print(f'\n------\nEach Block Contain {sum(ll)} Million Parameters (Based On {last_l} Block)')

    train_able_parameters = 0
    for i, (name, param) in enumerate(_model.named_parameters()):
        train_able_parameters += param.numel() / 1e6 if param.requires_grad else 0
    print(f'Total TrainAble Parameters In Model Is {train_able_parameters} Million Parameters')

    for i, (name, param) in enumerate(_model.named_parameters()):
        if param.requires_grad:
            print("{:>5} : {:<60} : {:>15} => {:>25}".format(i, name, param.requires_grad, param.numel() / 1e6))
    return _model

def collect_fn(batch):
    holder_dict = {}
    for key in batch[0].keys():
        tensor = torch.stack([torch.tensor(stack[key]) for stack in batch])
        holder_dict[key] = tensor
    return holder_dict
