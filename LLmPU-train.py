import argparse
import logging
import math
from typing import Tuple, Optional, Union

import erutils
import numpy as np
import pandas as pd
import torch
from erutils.loggers import show_hyper_parameters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import T5Tokenizer, AutoTokenizer

from config.config import TQDM_KWARGS

from modules.dataset import DatasetLLmPU
from modules.modeling_llmpu import LLmPUForConditionalGeneration, LLmPUConfig
from utils.utils import make2d, count_model_parameters, save_checkpoints, device_info, get_config_by_name, get_memory

logging.basicConfig(level=logging.WARN)
torch.backends.cudnn.benchmark = True
pars = argparse.ArgumentParser()
pars.add_argument('--batch-size', '--batch-size', type=int, default=1)
pars.add_argument('--epochs', '--epochs', type=int, default=100)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=False)
pars.add_argument('--load', '--load', type=bool, default=True)
pars.add_argument('--model', '--model', type=str, default='LLmPU-small')

opt = pars.parse_args()


def prepare_data(source_mask: Optional[torch.Tensor], source_ids: Optional[torch.Tensor],
                 target_ids: Optional[torch.Tensor], device: Union[torch.device, str]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    source_mask = make2d(source_mask)
    source_ids = make2d(source_ids)
    target_ids = make2d(target_ids)
    y = target_ids.to(device, dtype=torch.long)
    decoder_input: Optional[torch.Tensor] = y[:, :-1].contiguous()
    lm_labels: Optional[torch.Tensor] = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == 0] = -100
    input_id: Optional[torch.Tensor] = source_ids.to(device, dtype=torch.long)
    mask: Optional[torch.Tensor] = source_mask.to(device, dtype=torch.long)
    return input_id, mask, decoder_input, lm_labels


def train(m: Optional[LLmPUForConditionalGeneration],
          optim: Optional[torch.optim.AdamW],
          source_mask: Optional[torch.Tensor],
          source_ids: Optional[torch.Tensor],
          target_ids: Optional[torch.Tensor],
          device: Union[torch.device, str]) -> Optional[torch.Tensor]:
    input_ids, mask, decoder_input, labels = prepare_data(source_mask, source_ids, target_ids, device=device)
    out = m(input_ids=input_ids, attention_mask=mask, decoder_input_ids=decoder_input, labels=labels)
    loss_model = out[0]
    optim.zero_grad()
    loss_model.backward()
    optim.step()
    return loss_model


def _main(opt):
    device_info()
    board = SummaryWriter(log_dir=f'out/{opt.model}', filename_suffix=f'{opt.model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained('tokenizer_model/LLmPU')
    data_frame = pd.read_csv('ipynb/news_summary.csv')
    data_frame["text"] = "summarize: " + data_frame["text"]
    data_frame = data_frame[0:500]
    config: LLmPUConfig = get_config_by_name(opt.model, vocab_size=tokenizer.vocab_size)
    show_hyper_parameters(config)
    model = LLmPUForConditionalGeneration(config=config).to(device if not opt.load else 'cpu')
    erutils.fprint(f'Model Created with {count_model_parameters(model)} Million Parameters')

    source_length = config.max_length
    target_length = 32
    if opt.load:
        p_checkpoint = torch.load('LLmPU-model.pt')
        model.load_state_dict(p_checkpoint['model'])

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
        optimizer.load_state_dict(p_checkpoint['optimizer'])

        del p_checkpoint
    else:
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    dataset = DatasetLLmPU(tokenizer=tokenizer, source_len=source_length, target_len=target_length,
                           source_text=data_frame['text'], target_text=data_frame['headlines'])
    dataloader_kw = dict(batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    dataloader = DataLoader(dataset, **dataloader_kw)
    casual_iter = 0
    if opt.compile:
        model = torch.compile(model)
        erutils.fprint('Model Compiled Successfully !')
    mesh = config.mesh
    if opt.train:
        for epoch in range(opt.epochs):
            total_loss = 0
            with tqdm(iterable=enumerate(dataloader),
                      total=math.ceil(dataset.__len__() / opt.batch_size),
                      **TQDM_KWARGS) as progress_bar:
                for i, data in progress_bar:
                    casual_iter += 1

                    _source_ids, _source_mask, _target_ids = data['source_ids'], data['source_mask'], data['target_ids']
                    loss = train(model, optimizer, source_mask=_source_mask, source_ids=_source_ids,
                                 target_ids=_target_ids,
                                 device=device)
                    total_loss += loss
                    avg = total_loss.item() / (i + 1)
                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    progress_bar.set_postfix(loss=loss.item(), epoch=f'[{epoch}/{opt.epochs}]',
                                             avg=avg, free_GPU=free_gpu, used_GPU=used_gpu)
                    if (i + 1) % 50 == 0:
                        board_args = dict(global_step=casual_iter, new_style=True)
                        board.add_scalar('train/Loss', scalar_value=loss.item(), **board_args)
                        board.add_scalar('train/avgLoss', scalar_value=avg, **board_args)
                        board.add_scalar('train/epochs', scalar_value=epoch, **board_args)
                        board.add_scalar('train/gpu_used', scalar_value=used_gpu, **board_args)
                        board.add_scalar('train/gpu_free', scalar_value=free_gpu, **board_args)
                        board.add_scalars('cuda0', {
                            'total': total_gpu,
                            'free': free_gpu,
                            'used': used_gpu
                        }, global_step=casual_iter)
                        board.add_scalars('meshController', {
                            'sin': i * np.sin(i / mesh),
                            'cos': i * np.cos(i / mesh),
                            'tan': np.tan(i / mesh)
                        }, global_step=casual_iter)
                        board.add_scalar('train/meshIter_sin', scalar_value=i * np.sin(i / mesh), **board_args)
                        board.add_scalar('train/meshIter_cos', scalar_value=i * np.cos(i / mesh), **board_args)
                        board.add_scalar('train/meshIter_tan', scalar_value=np.tan(i / mesh), **board_args)
                progress_bar.write('=> Saving Model Checkpoints')
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=opt.epochs,
                                 epoch=epoch + 1,
                                 conf=config,
                                 config_name=opt.model,
                                 name='LLmPU-model.pt')


if __name__ == "__main__":
    _main(opt)
