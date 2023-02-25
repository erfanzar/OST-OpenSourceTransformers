import os
import typing

import torch
from erutils.loggers import fprint

from utils import DatasetPGT, get_config_by_name


# c = ''.join(f'{h}/' for h in os.getcwd().split('\\')[:-1])
# os.chdir(c)


def txt_2_pt(data_path: typing.Union[os.PathLike, str] = '../data/PGT-DATA-V2.txt'):
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    config = get_config_by_name('PGT-As')
    data = open(data_path, 'r', encoding="utf8").read()

    dataset = DatasetPGT(chunk=184, call_init=False, pt_data=False)
    selected_data = data
    dataset.src = selected_data
    tkn = dataset.tokenizer.encode_plus(
        text=dataset.src,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
        padding='do_not_pad',
        # max_length=self.chunk,
        truncation=False
    )['input_ids']
    chunk = 186
    v = torch.tensor([])
    for index in range(chunk, tkn.shape[1], chunk):
        q = tkn[:, index - chunk:index - 2]
        a = tkn[:, index - chunk + 1:index - 1]
        v = torch.cat([v, torch.cat([q, a], dim=-2).unsqueeze(0)],
                      dim=-3)
    print(f'VD : {v.shape}')
    torch.save(v, '../data/Data-conversation.pth')
    print('Saved Successfully')


if __name__ == "__main__":
    txt_2_pt()

