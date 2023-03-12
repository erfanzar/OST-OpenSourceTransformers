import argparse
import logging
import math
import typing
from typing import Optional, Union, Tuple

import erutils
import torch.utils.data
from datasets import load_dataset
from erutils.loggers import fprint
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, AutoTokenizer

from config.config import TQDM_KWARGS
from modules.dataset import DatasetLLmPChat, Tokens
from modules.models import LLmP, LLmPConfig
from utils.utils import make2d, save_checkpoints, get_config_by_name, device_info, get_memory, count_model_parameters

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--load', '--load', type=bool, default=True)
pars.add_argument('--model', '--model', type=str, default='LLmP-ML')
pars.add_argument('--out-path', '--out-path', type=str, default='out')
pars.add_argument('--data-src', '--data-src', type=str, default='data/convai.json')

options = pars.parse_args()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.CRITICAL)


def inter_q(question: Optional[str], tokenizer: GPT2Tokenizer, agent_name: str = '<LLmP> :') \
        -> Tuple[torch.Tensor, torch.Tensor]:
    out = tokenizer.encode_plus(Tokens.sos + question + agent_name, return_tensors='pt')
    return out['input_ids'], out['attention_mask']


def train(input_ids: Optional[Tensor],
          targets: Optional[Tensor],
          attention_mask: Optional[Tensor],
          network: Optional[LLmP.forward],
          optim: Optional[torch.optim.AdamW],
          loss_average: Optional[Tensor],
          device: Union[torch.device, str]) -> [typing.Union[torch.Tensor],
                                                typing.Union[torch.Tensor]]:
    labels: Optional[Tensor] = make2d(targets.type(torch.long).to(device))
    input_ids: Optional[Tensor] = make2d(input_ids.type(torch.long).to(device))
    logger.debug('RUNNING TRAIN FUNCTION IN MAIN THREAD ')
    _, loss = network(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    loss_average += loss.item()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return loss, loss_average


def main(opt):
    out_path = create_output_path(path=opt.out_path, name=opt.model)
    if not os.path.exists(os.path.join(out_path, 'weights')):
        os.mkdir(os.path.join(out_path, 'weights'))
    device_info()
    if opt.data_src.endswith('.txt'):
        data = open(opt.data_src, 'r', encoding='utf8').read().split()
    elif opt.data_src.endswith('.json'):
        data = opt.data_src
    elif opt.data_src.startswith('HF-'):
        name = opt.data_src.replace('HF-', '')
        if '/' in name:
            model_name = name.split('/')
            data = load_dataset(model_name[0], model_name[1])
        else:
            data = load_dataset(name)
        data = data["train"]['text']
        selected = int(len(data) * 0.01)
        data = data[:selected]
    else:
        data = None
        raise ValueError()
    parameters: LLmPConfig = get_config_by_name(opt.model)
    tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained('tokenizer_model/LLmP-C')

    dataset = DatasetLLmPChat(data=data, max_length=parameters.max_sentence_length, tokenizer=tokenizer)
    parameters.vocab_size = dataset.tokenizer.vocab_size

    parameters.vocab_size += 5
    # parameters.device = 'cpu'
    parameters.data_path = opt.data_src

    parameters.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=parameters.batch_size, num_workers=4,
                                             pin_memory=True)
    erutils.loggers.show_hyper_parameters(parameters)

    fprint('Loading Model ...' if opt.load else 'Creating Model ...')

    model = LLmP(config=parameters).to(parameters.device) if opt.load else LLmP(config=parameters).to('cpu')
    optimizer_kwargs = dict(lr=parameters.lr, weight_decay=parameters.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    model_parameters_size: typing.Optional[float] = count_model_parameters(model)

    checkpoints = torch.load(opt.weight, 'cpu') if opt.weight is not None else None

    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
        model = model.to(parameters.device)
        optimizer.load_state_dict(checkpoints['optimizer'])
    fprint(
        f'Model Loaded With {model_parameters_size} Million Parameters' if opt.load
        else f'Model Created With {model_parameters_size} Million Parameters')

    if opt.compile:
        model = torch.compile(model)
        fprint(f"Model Compiled Successfully")
    board = SummaryWriter(log_dir=f'{out_path}/tensorboard', filename_suffix=f'{opt.model}')
    at = 0

    question = 'Oh there you are'
    model = model.to(device=parameters.device)

    if opt.train:
        logger.info('TRAIN IS ABOUT TO START')
        for epoch in range(checkpoints['epoch'] if opt.load else 0, parameters.epochs):
            loss_avg = 0
            with tqdm(enumerate(dataloader), **TQDM_KWARGS,
                      total=math.ceil(dataset.__len__() // parameters.batch_size)) as progress_bar:
                for i, (input_ids_t, attention_mask) in progress_bar:
                    logger.debug(f'\033[1;94m input_ids_t    : {input_ids_t.shape}')
                    logger.debug(f'\033[1;94m attention_mask : {attention_mask.shape}')
                    # input_ids_t, attention_mask = input_ids_t.type_as(torch.float16), attention_mask.type_as(
                    #     torch.float16)
                    loss, loss_avg = train(input_ids=input_ids_t, targets=input_ids_t, network=model,
                                           optim=optimizer,
                                           loss_average=loss_avg, device=parameters.device,
                                           attention_mask=attention_mask)

                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    if ((i + 1) % 50) == 0:
                        tk, _ = inter_q(question, tokenizer=tokenizer
                                        )
                        tk = tk.to(parameters.device)
                        cals = []
                        for pred in model.generate(tokens=tk, pad_id=tokenizer.pad_token_id,
                                                   attention_mask=None,
                                                   eos_id=tokenizer.eos_token_id):
                            cals.append(pred)
                        cals = torch.cat(cals, dim=-1)
                        cals = cals.to('cpu')
                        awn = tokenizer.decode(cals[0])
                        del cals

                        board.add_scalar('train/Loss', scalar_value=loss.item(), global_step=at)
                        board.add_scalar('train/avg-Loss', scalar_value=(loss_avg / (i + 1)),
                                         global_step=at)
                        board.add_text('train/GeneratedResponse', f'question : {question} | answer : {awn}')
                    at += 1
                    progress_bar.set_postfix(epoch=f'[{epoch}/{parameters.epochs}]', device=parameters.device,
                                             loss_avg=(loss_avg / (i + 1)),
                                             loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)

                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=parameters.epochs,
                                 epoch=epoch + 1, config=opt.model,
                                 name=f'{out_path}/weights/{opt.model}-model.pt')
                progress_bar.write('==> MODEL SAVED SUCCESSFULLY')


if __name__ == "__main__":
    main(options)
