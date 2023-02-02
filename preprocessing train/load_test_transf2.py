from dataclasses import dataclass

import numpy as np
import sentencepiece
import torch.cuda
from torch.autograd import Variable
from torch.nn import functional as F

from transf2 import Transformer


@dataclass
class Conf:
    vocab = 1004
    embedded = 512
    heads = 4
    layers = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


transformer = Transformer(src_vocab=Conf.vocab,
                          trg_vocab=Conf.vocab,
                          d_model=Conf.embedded,
                          heads=Conf.heads,
                          N=Conf.layers).to(Conf.device)

# print(sum(s.numel() for s in transformer.parameters()) / 1e6, " Million Parameters Are In MODEL")
optim = torch.optim.AdamW(transformer.parameters(), 4e-4, betas=(0.9, 0.98), eps=1e-9)


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


se = sentencepiece.SentencePieceProcessor()
se.Load(model_file='../tokenizer_model/test_model.model')


def translate(model, src, max_len=200, custom_string=False):
    model.eval()
    # if custom_string == True:
    s = tokenize_words(se.Encode(src))
    print(s)
    src = torch.tensor(s).to(Conf.device).unsqueeze(0)

    src_mask = (src != 1003).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([0])
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
        # print(f'OUT : {out}')
        out = F.softmax(out, dim=-1)
        out = out[:, -1, :]
        # print(f'out = {out.shape}')

        ix = torch.multinomial(out, num_samples=1)
        print(ix)
        outputs[i] = ix[0]
        # print(outputs)
        if ix[0][0] == 1002:
            break
        # return outputs[:i]
    # print(outputs)
    return outputs


if __name__ == "__main__":
    print(transformer)
    state = torch.load('LAL.pt')
    print(*(k for k, v in state.items()))
    transformer.load_state_dict(state['model'])
    optim.load_state_dict(state['optimizer'])

    translate(transformer, 'what come after 1')
