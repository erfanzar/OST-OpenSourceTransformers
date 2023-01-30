import os
import sentencepiece as sen
import torch
import torch.nn as nn
import typing
from dataclasses import dataclass
from erutils.command_line_interface import fprint
from erutils.utils import read_json
from torch.nn import functional as F

import common_models as cm



def train(model, sentence: sen.SentencePieceProcessor):
    data = read_json(TrainConfig.data)
    for epoch in range(TrainConfig.epochs):
        for i, q_a_a in enumerate(data):
            x = tokenize_words(sentence.Encode(q_a_a['question']))
            target = tokenize_words(sentence.Encode(q_a_a['answers']))
            fprint('')


if __name__ == "__main__":
    se = sen.SentencePieceProcessor()
    se.Load(model_file='../tokenizer_model/test_model.model')
    decoder = cm.PTTDecoder(
        vocab_size=se.vocab_size(), n_layers=cm.Config.num_layer, n_head=cm.Config.num_head,
        n_embedded=cm.Config.n_embedded,
        head_size=cm.Config.head_size, chunk=cm.Config.chunk)

    text = 'Hello im the one Here Bras'
    # text = se.