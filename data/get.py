from erutils.utils import download

# data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
if __name__ == '__main__':
    download(data_url)
