from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 10000
    lr: float = 4e-4
    data: typing.Union[str, os.PathLike] = '../data/train-v2.0-cleared.json'


def tokenize_words(word: list, first_word_token: int = 0, last_word_token: int = 1002):
    """
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: 0 for start token | 1002 for end token
    """

    word = [first_word_token] + word
    word.append(last_word_token)
    return word


def detokenize_words(word: list, first_word_token: int = 0, last_word_token: int = 1002):
    """
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: un tokenized words
    """

    return [(first_word_token if w == last_word_token - 1 else w) for w in
            [w for w in word if w not in [last_word_token, first_word_token]]]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
