import torch.utils.data
from erutils.loggers import fprint

from modules.models import PGT
from utils.utils import DatasetPGTC, get_config_by_name


def _main():
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')

    Config = get_config_by_name('PGT-As')
    dataset = DatasetPGTC()
    Config.vocab_size = dataset.vocab_size
    Config.vocab_size += 2
    # Config.device = 'cpu'
    fprint('Loading Model ...')
    model = PGT(config=Config).to('cpu')
    loaded = torch.load('PGT-As-model.pt', 'cpu')
    model.load_state_dict(loaded['model'])
    model = model.to(Config.device)
    fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')

    print('🧠Let Have Conversation Dude')
    income = f'{dataset.sos}'
    model.eval()
    while True:
        income += input('>>> ')
        question = dataset.decode(dataset.encode(income)['input_ids'])
        text = dataset.encode(question)['input_ids'].to(Config.device)
        len_question = len(question.split())
        for p in range(200):
            text = model.generate_ca(text)
            wrt = dataset.decode(text)
            interface_response = ' '.join(k for k in wrt.split()[len_question:])
            print(f'\rAI : {interface_response}', end='')
            if text[0][-1] == dataset.tokenizer.eos_token_id:
                break
            if p % 50 == 0:
                print('\n')
        print()
        # break


if __name__ == "__main__":
    _main()
