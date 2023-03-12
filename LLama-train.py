import argparse
import logging
import math
import typing
from typing import Optional, Union

import erutils
import torch.utils.data
from datasets import load_dataset
from erutils.loggers import fprint
from torch import Tensor
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

from modules.dataset import DatasetLLama
from modules.modelling_llama import LLamaModel, LLamaConfig, Tokens
from utils.utils import make2d, save_checkpoints, get_config_by_name, device_info, get_memory

torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--out-path', '--out-path', type=str, default='out')

pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--model', '--model', type=str, default='LLama')
pars.add_argument('--data-src', '--data-src', type=str, default='HF-wikitext/wikitext-2-v1')

options = pars.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


def train(input_ids: Optional[Tensor],
          targets: Optional[Tensor],
          network: Optional[LLamaModel.forward],
          optim: Optional[torch.optim.AdamW],
          loss_function: Optional[torch.nn.CrossEntropyLoss],
          loss_average: Optional[Tensor],
          device: Union[torch.device, str]) -> [typing.Union[torch.Tensor],
                                                typing.Union[torch.Tensor]]:
    targets: Optional[Tensor] = make2d(targets.type(torch.long).to(device))
    input_ids: Optional[Tensor] = make2d(input_ids.type(torch.long).to(device))
    network.zero_grad(set_to_none=True)
    predict = network(tokens=input_ids, pos_start=0)

    # shift_logits = predict[..., :-1, :].contiguous()
    shift_logits = predict.contiguous()
    shift_labels = targets[..., -1].contiguous()

    shift_logits = shift_logits.view(-1 if shift_logits.shape[0] > 1 else 1, shift_logits.size(-1))

    shift_labels = shift_labels.view(-1)

    loss_prediction = loss_function(shift_logits, shift_labels)

    loss_average += loss_prediction.item()
    loss_prediction.backward()
    optim.step()
    return loss_prediction, loss_average


def main(opt):
    out_path = create_output_path(path=opt.out_path, name=opt.model)
    if not os.path.exists(os.path.join(out_path, 'weights')):
        os.mkdir(os.path.join(out_path, 'weights'))
    device_info()
    if not opt.data_src.startswith('HF-'):
        data = open(opt.data_src, 'r', encoding='utf8').read().split('<|endoftext|>')
    else:
        name = opt.data_src.replace('HF-', '')
        if '/' in name:
            model_name = name.split('/')
            data = load_dataset(model_name[0], model_name[1])
        else:
            data = load_dataset(name)
        data = data["train"]['text']
        selected = int(len(data) * 0.1)
        data = data[:selected]
    board = SummaryWriter(log_dir=f'{out_path}/tensorboard', filename_suffix=f'{opt.model}')

    parameters: LLamaConfig = get_config_by_name(opt.model)
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token=Tokens.eos,
                                                             pad_token=Tokens.pad, sos_token=Tokens.sos)
    dataset = DatasetLLama(data=data, max_length=parameters.max_sentence_length, tokenizer=tokenizer)
    parameters.vocab_size = dataset.tokenizer.vocab_size
    parameters.vocab_size += 2
    # parameters.device = 'cpu'
    parameters.data_path = opt.data_src

    parameters.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=parameters.batch_size, num_workers=4,
                                             pin_memory=True)
    erutils.loggers.show_hyper_parameters(parameters)

    fprint('Loading Model ...' if opt.weight is not None else 'Creating Model ...')

    model = LLamaModel(config=parameters).to(parameters.device) if opt.weight is not None else LLamaModel(
        config=parameters).to('cpu')
    optimizer_kwargs = dict(lr=parameters.lr, weight_decay=parameters.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    model_parameters_size: typing.Optional[float] = sum(p.numel() for p in model.parameters()) / 1e6

    checkpoints = torch.load(opt.weight, 'cpu') if opt.weight is not None else None

    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
        model = model.to(parameters.device)
        optimizer.load_state_dict(checkpoints['optimizer'])
    fprint(
        f'Model Loaded With {model_parameters_size} Million Parameters' if opt.weight is not None
        else f'Model Created With {model_parameters_size} Million Parameters')
    criterion = torch.nn.CrossEntropyLoss()

    if opt.compile:
        model = torch.compile(model)
        fprint(f"Model Compiled Successfully")

    question = dataset.encode(Tokens.sos + 'say something ').to(parameters.device)
    question = question['input_ids'].to(parameters.device)
    model = model.to(device=parameters.device)
    logger.info('TRAIN IS ABOUT TO START!!!')
    if opt.train:
        logger.info('TRAIN IS ABOUT TO START')
        at = 0
        for epoch in range(checkpoints['epoch'] if opt.load else 0, parameters.epochs):
            loss_avg = 0
            with tqdm(enumerate(dataloader), colour='blue',
                      total=math.ceil(dataset.__len__() // parameters.batch_size)) as progress_bar:
                for i, (input_ids_t) in progress_bar:
                    at += 1
                    loss, loss_avg = train(input_ids=input_ids_t, targets=input_ids_t, network=model, optim=optimizer,
                                           loss_average=loss_avg, loss_function=criterion, device=parameters.device)
                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    progress_bar.set_postfix(epoch=f'[{epoch}/{parameters.epochs}]', device=parameters.device,
                                             loss_avg=(loss_avg / (i + 1)),
                                             loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)
                    if (i + 1) % 50 == 0:
                        predictions = model.generate(prompts=question, max_gen_len=30,
                                                     pad_id=dataset.tokenizer.pad_token_id,
                                                     eos_id=dataset.tokenizer.eos_token_id)
                        board.add_scalar('train/Loss', scalar_value=loss.item(), global_step=at)
                        board.add_scalar('train/avg-Loss', scalar_value=(loss_avg / (i + 1)),
                                         global_step=at)
                        board.add_text('train/GeneratedResponse',
                                       f'QUESTION : {dataset.tokenizer.decode(question[0])} |'
                                       f' PREDICTION : {dataset.tokenizer.decode(predictions)}')

                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=parameters.epochs,
                                 epoch=epoch + 1, config=opt.model,
                                 name=f'{out_path}/weights/{opt.model}-model.pt')
                progress_bar.write('==> MODEL SAVED SUCCESSFULLY')


if __name__ == "__main__":
    main(options)
