from erutils.utils import read_json
import json

if __name__ == "__main__":
    data = read_json('../data/train-v2.0.json')
    total = {}

    for d in data['data']:
        for da in d['paragraphs']:

            for daa in da['qas']:
                # print([k for k, v in daa.items()])
                if 'question' in daa:
                    if daa['is_impossible'] is not True:
                        total[len(total)] = {'question': daa['question'], 'answers': daa['answers'][0]['text']}
                else:
                    for daaa in daa['qas']:
                        if 'question' in daaa:
                            if daaa['is_impossible'] is not True:
                                total[len(total)] = {'question': daaa['question'],
                                                     'answers': daaa['answers'][0]['text']}

    with open('train-v2.0-cleared.json', 'w') as stream:
        json.dump(total, stream)
    print('wrote successfully')
