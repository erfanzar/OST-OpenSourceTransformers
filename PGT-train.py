import argparse
import logging
import math
import os
import typing
from typing import Optional, Union, Tuple

import erutils
import torch.utils.data
from erutils.loggers import fprint
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from config.config import TQDM_KWARGS
from modules.dataset import DatasetPGTChat
from modules.models import PGT, Adafactor, PGTConfig
from utils.utils import make2d, save_checkpoints, get_config_by_name, device_info, get_memory, create_output_path, \
    get_data

logger = logging.getLogger(__name__)
Tensor = torch.Tensor
torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--out-path', '--out-path', type=str, default='out')
pars.add_argument('--model', '--model', type=str, default='PGT-S')
pars.add_argument('--data-src', '--data-src', type=str, default='data/convai.json')

options = pars.parse_args()


def inter_q(question: Optional[str], tokenizer) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    out = tokenizer.encode_plus(question, return_tensors='pt')
    return out['input_ids'], out['attention_mask']


def train(input_ids: Optional[Tensor],
          targets: Optional[Tensor],
          attention_mask: Optional[Tensor],
          network: Optional[PGT.forward],
          optim: Optional[Adafactor],
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
    task = 'Conversation :'
    if opt.weight is None:
        out_path = create_output_path(path=opt.out_path, name=opt.model)
        if not os.path.exists(os.path.join(out_path, 'weights')):
            os.mkdir(os.path.join(out_path, 'weights'))
    else:
        if opt.weight.endswith('.pt'):
            out_path = opt.weight.split('/')
            if 'weights' in out_path:
                out_path = os.path.join(*(p for p in out_path[:-2]))
            else:
                out_path = os.path.join(*(p for p in out_path[:-1]))
        else:
            raise ValueError('weight must contain path to .pt file')
    device_info()
    data = get_data(opt.data_src)
    config: PGTConfig = get_config_by_name(opt.model)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer_model/PGT-C')
    dataset = DatasetPGTChat(data=data, max_length=config.max_sentence_length, task=task, tokenizer=tokenizer)
    config.vocab_size = dataset.tokenizer.vocab_size
    config.vocab_size += 8
    config.device = 'cpu'
    config.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=4,
                                             pin_memory=True)
    erutils.loggers.show_hyper_parameters(config)

    model = PGT(config=config).to(config.device) if opt.weight is None else PGT(config=config).to('cpu')
    optimizer = Adafactor(model.parameters(), config.lr)
    model_parameters_size: typing.Optional[float] = sum(p.numel() for p in model.parameters()) / 1e6

    fprint('Loading Model ...' if opt.weight else 'Creating Model ...')

    checkpoints = torch.load(opt.weight, 'cpu') if opt.weight is not None else None

    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
        model = model.to(config.device)
        optimizer.load_state_dict(checkpoints['optimizer'])
        start_epoch = checkpoints['epoch']
        at = checkpoints['at']
        del checkpoints
    else:
        start_epoch = 0
        at = 0
    fprint(
        f'Model Loaded With {model_parameters_size} Million Parameters' if opt.weight is not None
        else f'Model Created With {model_parameters_size} Million Parameters')

    if opt.compile:
        #model = torch.compile(model)
        fprint(f"Model Compiled Successfully")
    if opt.train:
        board = SummaryWriter(log_dir=f'{out_path}/tensorboard', filename_suffix=f'{opt.model}')

    # question = task + '\n How do muscles grow?' + dataset.agent
    model = model.to(device=config.device)
    question = task + '\n hello how are you ?' + dataset.agent

    if opt.train:
        logger.info('TRAIN IS ABOUT TO START')
        for epoch in range(start_epoch, config.epochs):
            loss_avg = 0
            with tqdm(enumerate(dataloader), **TQDM_KWARGS,
                      total=math.ceil(dataset.__len__() // config.batch_size)) as progress_bar:
                for i, (input_ids_t, attention_mask) in progress_bar:
                    logger.debug(f'\033[1;94m input_ids_t    : {input_ids_t.shape}')
                    logger.debug(f'\033[1;94m attention_mask : {attention_mask.shape}')

                    loss, loss_avg = train(input_ids=input_ids_t, targets=input_ids_t, network=model,
                                           optim=optimizer,
                                           loss_average=loss_avg, device=config.device,
                                           attention_mask=attention_mask)

                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    if ((i + 1) % 50) == 0:
                        tk, _ = inter_q(question, tokenizer=dataset.tokenizer)
                        tk = tk.to(config.device)
                        cals = []
                        try:
                            for pred in model.generate(tokens=tk, pad_id=dataset.tokenizer.pad_token_id,
                                                       attention_mask=None,
                                                       eos_id=dataset.tokenizer.eos_token_id):
                                cals.append(pred)
                            cals = torch.cat(cals, dim=-1)
                            cals = cals.to('cpu')
                            awn = dataset.tokenizer.decode(cals[0])
                        except:
                            awn = 'EMPTY'
                        del cals

                        board.add_scalar('train/Loss', scalar_value=loss.item(), global_step=at)
                        board.add_scalar('train/avg-Loss', scalar_value=(loss_avg / (i + 1)),
                                         global_step=at)
                        board.add_text('train/Context', f'{question}', global_step=at)
                        board.add_text('train/GeneratedResponse', f'{awn}', global_step=at)
                    at += 1
                    progress_bar.set_postfix(epoch=f'[{epoch}/{config.epochs}]', device=config.device,
                                             loss_avg=(loss_avg / (i + 1)),
                                             loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)

                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=config.epochs, at=at,
                                 epoch=epoch + 1, config=opt.model,
                                 name=f'{out_path}/weights/{opt.model}-model.pt')
                progress_bar.write('==> MODEL SAVED SUCCESSFULLY')

    else:
        with tqdm(range(1), **TQDM_KWARGS,
                  total=1) as progress_bar:
            for i in progress_bar:
                (input_ids_t, attention_mask) = dataset.__getitem__(i)
                logger.debug(f'\033[1;94m input_ids_t    : {input_ids_t.shape}')
                logger.debug(f'\033[1;94m attention_mask : {attention_mask.shape}')

                loss, loss_avg = train(input_ids=input_ids_t, targets=input_ids_t, network=model,
                                       optim=optimizer,
                                       loss_average=torch.tensor(0.0).float(), device=config.device,
                                       attention_mask=attention_mask)

                free_gpu, used_gpu, total_gpu = get_memory(0)

                progress_bar.set_postfix(device=config.device,
                                         loss_avg=(loss_avg / (i + 1)),
                                         loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)


if __name__ == "__main__":
    main(options)
