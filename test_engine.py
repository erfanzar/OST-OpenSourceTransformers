import torch.utils.data
from erutils.utils import read_yaml, read_json
from modules.models import PGT
from utils.utils import create_config
from erutils.utils import read_json
from erutils.command_line_interface import fprint
import time
from utils.utils import DatasetPGT, make2d, save_model, get_config_by_name

if __name__ == "__main__":
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    data_path = 'data/q&a_cleaned.txt'

    dataset = DatasetPGT()
    Config = get_config_by_name('PGT-ss', dataset.vocab_size)
    Config.data_path = data_path
    dataset.chunk = Config.chunk
    data = open(Config.data_path, 'r').read()
    dataset.src = data
    Config.batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Config.batch_size, num_workers=2)
    fprint('Creating Model ...')
    model = PGT(config=Config).to(Config.device)
    fprint(f'Model Created With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
    total_iterations = dataset.__len__() // Config.batch_size
    question = dataset.encode('hello how are you ?').to(Config.device)
    # question_mask = question['attention_mask'].to(Config.device)
    question = question['input_ids'].to(Config.device)

    for epoch in range(Config.epochs):
        loss_avg = 0
        st = time.time()
        for i, inputs in enumerate(dataloader):
            x = inputs['x']
            y = inputs['y']
            # mask = inputs['mask']

            inp = make2d(x).to(Config.device)
            label = make2d(y).to(Config.device)
            # inp_mask = make2d(mask).to(Config.device)

            # print(inp.shape)
            # print(inp_mask.shape)
            # print(label.shape)
            predict = model(inputs=inp)
            # print(predict.shape)
            optimizer.zero_grad()
            loss = criterion(predict.view(-1, predict.size(-1)), label.view(-1))
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()
            fprint(
                f'\rEPOCH : [{epoch}/{Config.epochs}] | LOSS : {loss.item() / Config.batch_size} | EPOCH LOSS AVG : {(loss_avg / (i + 1)) / Config.batch_size} | ITER : {i + 1} | DEVICE : {Config.device} | EPOCH TIME {time.time() - st}',
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
            fprint(f'ANSWER   : {"Thank IM Doing Find"}')
            fprint(f'PREDICTION : {dataset.decode(predictions)}')
