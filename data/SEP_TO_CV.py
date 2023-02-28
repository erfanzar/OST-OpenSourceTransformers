import os
import typing


def pgt(path: typing.Union[os.PathLike, str] = 'PGT-DATA.txt'):
    data = open(path, 'r', encoding='utf8').read()
    dt = data.split(sep='[SEP]')
    vpp = 0
    string = ''
    use = False
    for i, idx in enumerate(range(0, len(dt), 2)):
        try:
            string += dt[idx] + '<|endoftext|>' + dt[idx + 1] + '<|endoftext|>'
        except:
            pass
    print(string)
    open('PGT-DATA-V2.txt', 'w', encoding='utf8').write(string)


def pgt_j(path: typing.Union[os.PathLike, str] = 'PGT-DATA.txt'):
    data = open(path, 'r', encoding='utf8').read()
    dt = data.split(sep='[SEP]')
    string = ' '.join(d + '<|endoftext|>' for d in dt)
    print(string)
    open('PGT-J-DATA.txt', 'w', encoding='utf8').write(string)


if __name__ == "__main__":
    pgt()
