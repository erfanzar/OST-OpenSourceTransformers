import erutils
import sentencepiece


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


sentence = sentencepiece.SentencePieceProcessor()
sentence.Load(model_file='../tokenizer_model/test_model.model')

data = erutils.read_json('../data/train-v2.0-cleared.json')


def fix_data(data):
    for d in data:
        question = data[d]['question']
        answers = data[d]['answers']
        encoded_question = tokenize_words(sentence.Encode(question))
        encoded_answers = tokenize_words(sentence.Encode(answers))
        yield encoded_question, encoded_answers


def add_tokens(text):
    sos = '<SOS>'
    eos = '<EOS>'
    pad = '<PAD>'
    return sos + text + eos + pad


if __name__ == '__main__':
    for d in fix_data(data):
        print(d)
