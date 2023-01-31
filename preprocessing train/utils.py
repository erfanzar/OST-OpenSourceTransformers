from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 10000
    lr: float = 4e-4
    data: typing.Union[str, os.PathLike] = '../data/train-v2.0-cleared.json'

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
