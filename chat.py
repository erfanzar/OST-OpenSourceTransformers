import torch.utils.data
from erutils.loggers import fprint

from modules.models import PGT
from utils.utils import DatasetPGT, get_config_by_name

if __name__ == "__main__":
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    dataset = DatasetPGT(call_init=False, pt_data=False)
    Config = get_config_by_name('PGT-As', dataset.vocab_size)
    Config.device = 'cpu'
    dataset.chunk = Config.chunk
    fprint('Loading Model ...')
    model = PGT(config=Config).to('cpu')
    loaded = torch.load('PGT-As-model.pt', 'cpu')
    model.load_state_dict(loaded['model'])
    model = model.to(Config.device)
    fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
    # optimizer.load_state_dict(loaded['optimizer'])
    print('🧠Let Have Conversation Dude')
    income = ''
    while True:
        income += input('>>> ')
        # income = 'hello how are you today ?'
        question = dataset.decode(dataset.encode(income)['input_ids'])
        text = dataset.encode(question)['input_ids'].to(Config.device)
        len_question = len(question.split())
        for p in range(200):
            text = model.generate_ca(text)
            wrt = dataset.decode(text)
            interface_response = ' '.join(k for k in wrt.split()[len_question:])
            print(f'\rAI : {interface_response}', end='')
            if text[0][-1] == 102:
                break
            if p % 50 == 0:
                print('\n')
        print()
        # break
