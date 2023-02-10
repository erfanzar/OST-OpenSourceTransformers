import torch.utils.data
from erutils.utils import read_yaml, read_json
from modules.models import PGT
from utils.utils import create_config
from erutils.utils import read_json
from erutils.command_line_interface import fprint
from utils.utils import DatasetQA, make2d, save_model

if __name__ == "__main__":
    Config = create_config(
        batch_size=6,
        data_path='data/q&a.json',
        num_heads=8,
        chunk=512,
        num_embedding=512,
        num_layers=6
    )
    data = read_json(Config.data_path)
    src, trg = [data[s]['question'] for s in data], [data[s]['answer'] for s in data]
    dataset = DatasetQA(src=src, trg=trg, max_length=Config.max_position_embeddings)
    Config.vocab_size = dataset.tokenizer.vocab_size
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Config.batch_size, num_workers=2)
    fprint('Creating Model ...')
    model = PGT(config=Config)
    fprint(f'Model Created With {sum(p.numel() for p in model.parameters()) / 1e6} Million Parameters')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), Config.lr)
    total_iterations = dataset.__len__() // Config.batch_size
    data_ip = dataset.__getitem__(1)
    answer = data_ip['input']['input_ids'].to(Config.device)
    question = data_ip['label']['input_ids'].to(Config.device)
    for epoch in range(Config.epochs):
        loss_avg = 0
        for i, inputs in enumerate(dataloader):
            inp = make2d(inputs['input']['input_ids'])
            inp_mask = make2d(inputs['input']['attention_mask'])
            label = make2d(inputs['label']['input_ids'])
            # label_mask = make2d(inputs['label']['attention_mask'])
            predict = model(inputs=inp, attention_mask=inp_mask)
            # print(predict.shape)
            optimizer.zero_grad()
            loss = criterion(predict.permute(0, 2, 1), label)
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()
            fprint(
                f'\rEPOCH : [{epoch}/{Config.epochs}] | LOSS : {loss.item() / Config.batch_size} | EPOCH LOSS AVG : {(loss_avg / (i + 1)) / Config.batch_size} | ITER : {i + 1}',
                end='')

        print()
        if epoch % 5 == 0:
            print()
            save_model(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=Config.epochs, epoch=epoch,
                       name='model.pt')
            fprint('==> MODEL SAVE SUCCESSFULLY')
            predictions = model.generate(idx=question, eos=dataset.tokenizer.eos_token_id)
            fprint(f'QUESTION : {dataset.decode(question)}')
            fprint(f'ANSWER   : {dataset.decode(answer)}')
            fprint(f'PREDICTION : {dataset.decode(predictions)}')
