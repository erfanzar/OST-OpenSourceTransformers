import argparse
import math
import time
import typing

import erutils
import torch.utils.data
from erutils.loggers import fprint

from modules.models import PGT
from utils.utils import DatasetPGT, make2d, save_checkpoints, get_config_by_name, device_info

torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=16)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--load', '--load', type=bool, default=False)
pars.add_argument('--model', '--model', type=str, default='PGT-As')
pars.add_argument('--data-src', '--data-src', type=str, nargs='+', default=['data/Data-conversation.pth'])

options = pars.parse_args()


def main(opt):
    def train(src, targets, network, optim, loss_function, loss_average, device) -> [typing.Union[torch.Tensor],
                                                                                     typing.Union[torch.Tensor]]:
        targets = make2d(targets.type(torch.long)).to(device)
        predict = network(inputs=make2d(src.type(torch.long)).to(device))
        optim.zero_grad(set_to_none=True)
        loss_prediction = loss_function(predict.permute(0, 2, 1), targets.view(-1, targets.size(-1))) * targets.size()[
            0]
        loss_average += loss_prediction.item()
        loss_prediction.backward()
        optim.step()
        return loss_prediction, loss_average

    device_info()

    dataset = DatasetPGT(batch_size=opt.batch, pt_data=True, src=opt.data_src)

    parameters = get_config_by_name(opt.model, dataset.vocab_size)

    parameters.data_path = opt.data_src
    dataset.chunk = parameters.chunk

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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if opt.compile:
        model = torch.compile(model)
        fprint(f"Model Compiled Successfully")

    question = dataset.encode('USER: hello how are you ?').to(parameters.device)
    question = question['input_ids'].to(parameters.device)
    mxl = math.ceil(dataset.__len__() / parameters.batch_size)

    if opt.train:
        for epoch in range(checkpoints['epoch'] if opt.load else 0, parameters.epochs):
            loss_avg = 0
            st = time.time()
            for i, (inp, label) in enumerate(dataloader):
                loss, loss_avg = train(src=inp, targets=label, network=model, optim=optimizer,
                                       loss_average=loss_avg, loss_function=criterion, device=parameters.device)
                fprint(
                    f'\rEPOCH : [{epoch + 1}/{parameters.epochs}] | LOSS : {loss.item() / parameters.batch_size} |'
                    f' EPOCH LOSS AVG : {(loss_avg / (i + 1)) / parameters.batch_size} | ITER : {i + 1}/{mxl} |'
                    f' DEVICE : {parameters.device} | EPOCH TIME {int(time.time() - st)} SEC',
                    end='')

            if (epoch + 1) % 5 == 0:
                print()
                save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=parameters.epochs,
                                 epoch=epoch + 1, config=opt.model,
                                 name='model.pt')
                fprint('==> MODEL SAVED SUCCESSFULLY')
                predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id,
                                             generate=256

                                             )
                fprint(f'QUESTION : {dataset.decode(question)}')
                fprint(f'PREDICTION : {dataset.decode(predictions)}')


if __name__ == "__main__":
    main(options)
