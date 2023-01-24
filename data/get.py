from erutils.utils import download

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if __name__ == '__main__':
    download(data_url)
