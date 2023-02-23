import math
import time
import typing

import torch.utils.data
from erutils.loggers import fprint

from modules.models import PGT
from utils.utils import DatasetPGT, make2d, save_model, get_config_by_name
import argparse

torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=16)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--load', '--load', type=bool, default=False)
pars.add_argument('--model', '--model', type=str, default='PGT-As')
pars.add_argument('--data-src', '--data-src', type=typing.Union[str, list[str]], default=['data/Data-conversation.pth'])

options = pars.parse_args()


def main(opt):
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')

    dataset = DatasetPGT(batch_size=opt.batch, pt_data=True, src=opt.data_src)

    Config = get_config_by_name(opt.model, dataset.vocab_size)
    Config.load = opt.load
    Config.train = opt.train
    Config.data_path = opt.data_src
    dataset.chunk = Config.chunk

    Config.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Config.batch_size, num_workers=4,
                                             pin_memory=True)
    fprint('Loaded Configs :: =>')
    for d in Config.__dict__:
        try:
            print('{:<25} : {:>25}'.format(d, Config.__dict__[d]))
        except:
            pass
    if Config.load:
        fprint('Loading Model ...')
        model = PGT(config=Config).to('cpu')
        loaded = torch.load('model.pt', 'cpu')
        model.load_state_dict(loaded['model'])
        model = model.to(Config.device)
        fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = model.configure_optimizer(Config)
        optimizer.load_state_dict(loaded['optimizer'])
    else:
        fprint('Creating Model ...')
        model = PGT(config=Config).to('cpu').to(Config.device)
        fprint(f'Model Created With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = model.configure_optimizer(Config)

    model = torch.compile(model)

    total_iterations = dataset.__len__() // Config.batch_size
    question = dataset.encode('USER: hello how are you ?').to(Config.device)
    question = question['input_ids'].to(Config.device)
    mxl = math.ceil(dataset.__len__() / Config.batch_size)
    print('TRAINING IS ABOUT TO START')
    if Config.train:
        if Config.load:
            for epoch in range(loaded['epoch'], Config.epochs):
                loss_avg = 0
                st = time.time()
                for i, (inp, label) in enumerate(dataloader):
                    inp = inp.type(torch.long)
                    label = label.type(torch.long)
                    inp = make2d(inp).to(Config.device)
                    label = make2d(label).to(Config.device)
                    predict = model(inputs=inp)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(predict.permute(0, 2, 1), label.view(-1, label.size(-1)))
                    loss_avg += loss.item()
                    loss.backward()
                    optimizer.step()
                    fprint(
                        f'\rEPOCH : [{epoch + 1}/{Config.epochs}] | LOSS : {loss.item() / Config.batch_size} | EPOCH LOSS AVG : {(loss_avg / (i + 1)) / Config.batch_size} | ITER : {i + 1}/{mxl} | DEVICE : {Config.device} | EPOCH TIME {int(time.time() - st)} SEC',
                        end='')

                print()
                if (epoch + 1) % 5 == 0:
                    print()
                    save_model(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=Config.epochs,
                               epoch=epoch + 1,
                               name='modified_model.pt')
                    fprint('==> MODEL SAVED SUCCESSFULLY')
                    predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id,
                                                 generate=256

                                                 )
                    fprint(f'QUESTION : {dataset.decode(question)}')
                    fprint(f'PREDICTION : {dataset.decode(predictions)}')
        else:
            for epoch in range(Config.epochs):
                loss_avg = 0
                st = time.time()
                for i, (inp, label) in enumerate(dataloader):
                    inp = inp.type(torch.long)
                    label = label.type(torch.long)
                    inp = make2d(inp).to(Config.device)
                    label = make2d(label).to(Config.device)
                    predict = model(inputs=inp)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(predict.permute(0, 2, 1), label.view(-1, label.size(-1)))
                    loss_avg += loss.item()
                    loss.backward()
                    optimizer.step()
                    fprint(
                        f'\rEPOCH : [{epoch + 1}/{Config.epochs}] | LOSS : {loss.item() / Config.batch_size} | EPOCH LOSS AVG : {(loss_avg / (i + 1)) / Config.batch_size} | ITER : {i + 1}/{mxl} | DEVICE : {Config.device} | EPOCH TIME {int(time.time() - st)} SEC',
                        end='')

                print()
                if (epoch + 1) % 5 == 0:
                    print()
                    save_model(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=Config.epochs,
                               epoch=epoch + 1,
                               name='model.pt')
                    fprint('==> MODEL SAVED SUCCESSFULLY')
                    predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id,
                                                 generate=256

                                                 )
                    fprint(f'QUESTION : {dataset.decode(question)}')
                    fprint(f'PREDICTION : {dataset.decode(predictions)}')


if __name__ == "__main__":
    main(options)
