import os
import typing
from dataclasses import dataclass
import sentencepiece as sen
import torch
import torch.nn as nn
from erutils.command_line_interface import fprint
from torch.nn import functional as F
import common_models as cm
from erutils.utils import read_json


@dataclass
class TrainConfig:
    epochs: int = 10000
    lr: float = 4e-4
    data: typing.Union[str, os.PathLike] = '../data/train-v2.0-cleared.json'


def tokenize_words(word: list):
    """
    :param word: index
    :return: 0 for start token | 1002 for end token
    """

    word = [0] + word
    word.append(1002)
    return word


def detokenize_words(word: list):
    """
    :param word: index
    :return: un tokenized words
    """

    return [(0 if w == 1001 else w) for w in [w for w in word if w not in [1002, 0]]]


def train(model, sentence: sen.SentencePieceProcessor):
    data = read_json(TrainConfig.data)
    for epoch in range(TrainConfig.epochs):
        for i, q_a_a in enumerate(data):
            x = tokenize_words(sentence.Encode(q_a_a['question']))
            target = tokenize_words(sentence.Encode(q_a_a['answers']))
            fprint('')

if __name__ == "__main__":
    se = sen.SentencePieceProcessor()
    decoder = cm.PTTDecoder(
        vocab_size=se.vocab_size(), n_layers=cm.Config.num_layer, n_head=cm.Config.num_head,
        n_embedded=cm.Config.n_embedded,
        head_size=cm.Config.head_size, chunk=cm.Config.chunk)

    se.Load(model_file='../tokenizer_model/test_model.model')
