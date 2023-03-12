import argparse
import math
import typing
from typing import Optional, Union

import erutils
import torch.utils.data
from datasets import load_dataset
from erutils.loggers import fprint
from tqdm.auto import tqdm

from modules.models import PGT
from utils.utils import DatasetPGTC, make2d, save_checkpoints, get_config_by_name, device_info, get_memory

Tensor = torch.Tensor
torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=2)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--model', '--model', type=str, default='PGT-As')
pars.add_argument('--data-src', '--data-src', type=str, default='HF-wikitext/wikitext-103-raw-v1')

options = pars.parse_args()


def main(opt):
    def train(input_ids: Optional[Tensor],
              targets: Optional[Tensor],
              attention_mask: Optional[Tensor],
              network: Optional[PGT],
              optim: Optional[torch.optim.AdamW],
              loss_function: Optional[torch.nn.CrossEntropyLoss],
              loss_average: Optional[Tensor],
              device: Union[torch.device, str]) -> [typing.Union[torch.Tensor],
                                                    typing.Union[torch.Tensor]]:
        targets: Optional[Tensor] = make2d(targets.type(torch.long).to(device))
        input_ids: Optional[Tensor] = make2d(input_ids.type(torch.long).to(device))
        attention_mask: Optional[Tensor] = make2d(attention_mask.to(device))
        predict = network(inputs=input_ids,
                          attention_mask=attention_mask)
        optim.zero_grad(set_to_none=True)

        shift_logits = predict[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        loss_prediction = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss_average += loss_prediction.item()
        loss_prediction.backward()
        optim.step()
        return loss_prediction, loss_average

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
    parameters = get_config_by_name(opt.model)
    dataset = DatasetPGTC(data=data, chunk=parameters.chunk)
    parameters.vocab_size = dataset.vocab_size
    parameters.vocab_size += 2
    # parameters.device = 'cpu'
    parameters.data_path = opt.data_src

    parameters.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=parameters.batch_size, num_workers=4,
                                             pin_memory=True)
    erutils.loggers.show_hyper_parameters(parameters)

    fprint('Loading Model ...' if opt.load else 'Creating Model ...')

    model = PGT(config=parameters).to(parameters.device) if opt.load else PGT(config=parameters).to('cpu')
    optimizer = model.configure_optimizer(parameters)
    model_parameters_size: typing.Optional[float] = sum(p.numel() for p in model.parameters()) / 1e6

    checkpoints = torch.load('model.pt', 'cpu') if opt.load else None

    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
        model = model.to(parameters.device)
        optimizer.load_state_dict(checkpoints['optimizer'])
    fprint(
        f'Model Loaded With {model_parameters_size} Million Parameters' if opt.load
        else f'Model Created With {model_parameters_size} Million Parameters')
    criterion = torch.nn.CrossEntropyLoss()

    if opt.compile:
        model = torch.compile(model)
        fprint(f"Model Compiled Successfully")

    question = dataset.encode('USER: hello how are you ?').to(parameters.device)
    question = question['input_ids'].to(parameters.device)
    model = model.to(device=parameters.device)
    if opt.train:

        for epoch in range(checkpoints['epoch'] if opt.load else 0, parameters.epochs):
            loss_avg = 0
            with tqdm(enumerate(dataloader), colour='white',
                      total=math.ceil(dataset.__len__() // parameters.batch_size)) as progress_bar:
                for i, (input_ids_t, attention_mask_t) in progress_bar:
                    loss, loss_avg = train(input_ids=input_ids_t, targets=input_ids_t, network=model, optim=optimizer,
                                           attention_mask=attention_mask_t,
                                           loss_average=loss_avg, loss_function=criterion, device=parameters.device)
                    free_gpu, used_gpu, total_gpu = get_memory(0)
                    progress_bar.set_postfix(epoch=f'[{epoch}/{parameters.epochs}]', device=parameters.device,
                                             loss_avg=(loss_avg / (i + 1)),
                                             loss=loss.item(), free_GPU=free_gpu, used_GPU=used_gpu)

                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epochs=parameters.epochs,at=at,
                                 epoch=epoch + 1, config=opt.model,
                                 name='model.pt')
                progress_bar.write('==> MODEL SAVED SUCCESSFULLY')
                predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id,
                                             generate=256

                                             )
                progress_bar.write(f'QUESTION : {dataset.decode(question)}')
                progress_bar.write(f'PREDICTION : {dataset.decode(predictions)}')


if __name__ == "__main__":
    main(options)
