def main():
    data = open('PGT.txt', 'r', encoding='utf8').read()
    data = data.split('<|endoftext|>')
    f = ''
    print(data)
    for i in range(0, len(data), 2):
        try:
            f += '<|endoftext|>' + data[i+1] + data[i + 2]
        except:
            pass
    open('TPAP.txt', 'w', encoding='utf8').write(f)


if __name__ == '__main__':
    main()
