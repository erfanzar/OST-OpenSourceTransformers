import torch.utils.data
from erutils.loggers import fprint
from transformers import AutoTokenizer

from modules.dataset import DatasetLLmP, Tokens
from modules.models import LLmP
from utils.utils import get_config_by_name


def _main():
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')

    config = get_config_by_name('LLmP')
    tokenizer = AutoTokenizer.from_pretrained('tokenizer_model', bos_token=Tokens.eos,
                                              pad_token=Tokens.pad, sos_token=Tokens.sos)
    dataset = DatasetLLmP(data=[], tokenizer=tokenizer)
    config.vocab_size = dataset.tokenizer.vocab_size
    config.vocab_size += 1
    # config.device = 'cpu'
    fprint('Loading Model ...')
    model: LLmP = LLmP(config=config).to('cpu')
    loaded = torch.load('LLmP-model.pt', 'cpu')
    model.load_state_dict(loaded['model'])
    model = model.to(config.device)
    fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')

    print('🧠Let Have Conversation Dude')
    income = f'{tokenizer.eos_token}'

    model.eval()
    while True:
        income = input('>>> ')
        text = tokenizer.encode(Tokens.eos + income + Tokens.eos, return_tensors='pt').to(config.device)

        for v in model.generate(text, 240):
            print(f'{tokenizer.decode(v[0], skip_special_tokens=True)}', end='')
            if v[0] == tokenizer.eos_token_id:
                break
        print()


if __name__ == "__main__":
    _main()
