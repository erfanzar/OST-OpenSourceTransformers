import os
import typing

import torch
from erutils.command_line_interface import fprint

from utils import DatasetPGT, get_config_by_name


# c = ''.join(f'{h}/' for h in os.getcwd().split('\\')[:-1])
# os.chdir(c)


def txt_2_pt(percentage: float = 0.5, data_path: typing.Union[os.PathLike, str] = '../data/PGT-DATA.txt'):
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    config = get_config_by_name('PGT-As')
    dataset = DatasetPGT(chunk=config.chunk)
    data = open(data_path, 'r', encoding="utf8").read()
    tvl = len(data)
    use_tvl = tvl * percentage
    print(f'TOTAL DATA : {tvl}')
    print(f'SELECTED DATA : {int(use_tvl)}')
    selected_data = data[int(use_tvl):]
    dataset.src = selected_data
    with open('selected.txt', 'w', encoding='utf8') as wr:
        wr.write(selected_data)
    dataset.init()
    torch.save(dataset.data, '../data/Data-part-2.pt')
    print('Saved Successfully')


if __name__ == "__main__":
    txt_2_pt()
    # data = torch.load('Data.pt')
    # print(data.shape)
