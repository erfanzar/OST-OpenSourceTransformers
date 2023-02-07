import json

if __name__ == "__main__":
    v = {}
    with open('q&a.txt', 'r') as stream:
        data = stream.readlines()
        for i, line in enumerate(data):
            data_list = line.split('\t')
            v[i] = {
                'question': data_list[0],
                'answer': data_list[1]
            }

    with open('q&a.json', 'w') as stream:
        json.dump(v, stream)
