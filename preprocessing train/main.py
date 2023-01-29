import sentencepiece as sen
import torch
import torch.nn as nn
from torch.nn import functional as F
import common_models as cm
from erutils.utils import read_json


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


if __name__ == "__main__":
    # decoder = cm.PTTDecoder(
    #     vocab_size=64, n_layers=cm.Config.num_layer, n_head=cm.Config.num_head, n_embedded=cm.Config.n_embedded,
    #     head_size=cm.Config.head_size, chunk=cm.Config.chunk)
    se = sen.SentencePieceProcessor()
    se.Load(model_file='../tokenizer_model/test_model.model')

