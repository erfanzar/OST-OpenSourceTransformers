if __name__ == "__main__":
    data = open('PGT-DATA.txt', 'r', encoding='utf8').read()
    dt = data.split(sep='[SEP]')
    vpp = 0
    string = ''
    use = False
    for i, idx in enumerate(range(0, len(dt), 2)):
        try:
            string += ' USER:' + dt[idx] + ' PGT:' + dt[idx + 1]
        except:
            pass
    print(string)
    open('PGT-DATA-V2.txt', 'w', encoding='utf8').write(string)
