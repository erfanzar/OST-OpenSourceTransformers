if __name__ == "__main__":
    data = open('PGT-DATA.txt', 'r', encoding='utf8').read()
    dt = data.split(sep='[SEP]')
    vpp = 0
    string = ''
    use = False
    for i, d in enumerate(dt):
        try:
            if use:
                string += 'USER:' + d + 'PGT:' + dt[i + 1]
            else:
                use = True
        except:
            pass
    print(string)
    open('PGT-DATA-V2.txt', 'w', encoding='utf8').write(string)
