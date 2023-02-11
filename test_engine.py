import torch.utils.data
from erutils.utils import read_yaml, read_json
from modules.models import PGT
from utils.utils import create_config
from erutils.utils import read_json
from erutils.command_line_interface import fprint
import time
import math
from utils.utils import DatasetPGT, make2d, save_model, get_config_by_name

if __name__ == "__main__":
    batch = 4
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')

    data_path = 'data/q&a_cleaned.txt'
    dataset = DatasetPGT(batch_size=batch)

    Config = get_config_by_name('PGT-ss', dataset.vocab_size)
    Config.load = False

    Config.data_path = data_path
    dataset.chunk = Config.chunk

    data = open(Config.data_path, 'r').read()
    dataset.src = data

    Config.batch_size = batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Config.batch_size, num_workers=2)

    if Config.load:
        fprint('Loading Model ...')
        model = PGT(config=Config).to('cpu')
        loaded = torch.load('model.pt', 'cpu')
        model.load_state_dict(loaded['model'])
        model = model.to(Config.device)
        fprint(f'Model Loaded With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
        optimizer.load_state_dict(loaded['optimizer'])
    else:
        fprint('Creating Model ...')
        model = PGT(config=Config).to('cpu').to(Config.device)
        fprint(f'Model Created With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.AdamW(model.parameters(), Config.lr)

    total_iterations = dataset.__len__() // Config.batch_size
    question = dataset.encode('hello how are you ?').to(Config.device)
    question = question['input_ids'].to(Config.device)
    mxl = math.ceil(dataset.__len__() / Config.batch_size)
    for epoch in range(Config.epochs):
        loss_avg = 0
        st = time.time()
        # for i, inputs in enumerate(dataloader):
        for i, (x, y) in enumerate(dataloader):
            # x = inputs['x']
            # y = inputs['y']

            inp = make2d(x).to(Config.device)
            label = make2d(y).to(Config.device)
            # inp_mask = make2d(mask).to(Config.device)
            predict = model(inputs=inp)
            # print(predict.shape)
            optimizer.zero_grad()
            loss = criterion(predict.permute(0, 2, 1), label.view(-1, label.size(-1)))
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()
            fprint(
                f'\rEPOCH : [{epoch}/{Config.epochs}] | LOSS : {loss.item() / Config.batch_size} | EPOCH LOSS AVG : {(loss_avg / (i + 1)) / Config.batch_size} | ITER : {i + 1}/{mxl} | DEVICE : {Config.device} | EPOCH TIME {time.time() - st}',
                end='')

        print()
        if epoch % 15 == 0:
            print()
            save_model(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=Config.epochs, epoch=epoch,
                       name='model.pt')
            fprint('==> MODEL SAVED SUCCESSFULLY')
            predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id,
                                         # attention_mask=question_mask
                                         )
            fprint(f'QUESTION : {dataset.decode(question)}')
            fprint(f'ANSWER   : {"Thank IM Doing Fine"}')
            fprint(f'PREDICTION : {dataset.decode(predictions)}')
