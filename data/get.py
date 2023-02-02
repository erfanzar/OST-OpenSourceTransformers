from erutils.utils import download

# data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
pt = 'https://download.pytorch.org/models/text/xlmr.vocab.pt'
model = "https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
if __name__ == '__main__':
    download(data_url)
