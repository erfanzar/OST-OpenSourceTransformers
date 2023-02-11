import torch.utils.data
from erutils.utils import read_yaml, read_json
from modules.models import PGT
from utils.utils import create_config
from erutils.utils import read_json
from erutils.command_line_interface import fprint
import time
import math
from utils.utils import DatasetPGT, make2d, save_model, get_config_by_name

if __name__ == "__main__":
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    data_path = 'data/q&a_cleaned.txt'
    dataset = DatasetPGT()
    Config = get_config_by_name('PGT-ss', dataset.vocab_size)
    Config.data_path = data_path
    dataset.chunk = Config.chunk
    fprint('Loading Model ...')
    model = PGT(config=Config).to('cpu')
    loaded = torch.load('model.pt', 'cpu')
    model.load_state_dict(loaded['model'])
    model = model.to(Config.device)
    fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
    # optimizer.load_state_dict(loaded['optimizer'])
    print('🧠Let Have Conversation Dude')
    while True:
        text = dataset.encode(input('>>> '))['input_ids'].to(Config.device)
        for _ in range(200):
            text = model.generate_ca(text)
            wrt = dataset.decode(text)
            print(f'\rAI : {wrt}', end='')

            if text[0][-1] == 102:

                break
        print()
