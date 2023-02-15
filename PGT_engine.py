import math
import time

import torch.utils.data
from erutils.command_line_interface import fprint

from modules.models import PGT
from utils.utils import DatasetPGT, make2d, save_model, get_config_by_name

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    batch = 8
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')

    data_path = ['data/Data-part-1.pt', 'data/Data-part-2.pt']
    dataset = DatasetPGT(batch_size=batch, pt_data=True)

    Config = get_config_by_name('PGT-As', dataset.vocab_size)
    Config.load = False
    Config.train = True
    Config.data_path = data_path
    dataset.chunk = Config.chunk

    Config.batch_size = batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Config.batch_size, num_workers=4,
                                             pin_memory=True)

    if Config.load:
        fprint('Loading Model ...')
        model = PGT(config=Config).to('cpu')
        loaded = torch.load('model.pt', 'cpu')
        model.load_state_dict(loaded['model'])
        model = model.to(Config.device)
        fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = model.configure_optimizer(Config)
        # optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
        # optimizer = model.configure_optimizer(Config)
        optimizer.load_state_dict(loaded['optimizer'])
    else:
        fprint('Creating Model ...')
        model = PGT(config=Config).to('cpu').to(Config.device)
        fprint(f'Model Created With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = model.configure_optimizer(Config)
        # optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
    model = torch.compile(model)

    total_iterations = dataset.__len__() // Config.batch_size
    question = dataset.encode('what do you know about dubai').to(Config.device)
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


