import math

import torch.utils.data
from rwkv.model import RWKV
from modules import RWKVConfig, RWKV_GPT_CasualLM, RWKVConfigTrain
from utils.utils import count_model_parameters, get_data, save_checkpoints
from modules.datasets import CasualLMDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from erutils.utils import create_output_path, make2d
from torch.utils.tensorboard import SummaryWriter

config = RWKVConfig(
    hidden_size=512,
    number_of_layers=8,
    vocab_size=32000,
)
train_config = RWKVConfigTrain()


def train():
    CTX_LEN = 128
    device = config.device
    data = get_data('data/alpaca_data.json')[:5000]
    model = RWKV_GPT_CasualLM(config=config).to(device)
    optimizer = model.configure_optimizers(train_config)
    model = torch.jit.script(model)

    print(f'Model Contain : {count_model_parameters(model)} M Parameters')
    tokenizer = AutoTokenizer.from_pretrained('tokenizer_model/BASE')
    dataset = CasualLMDataset(tokenizer=tokenizer, data=data, max_length=CTX_LEN)
    out_path = create_output_path(name='RWKV', path='out')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    logdir = str(out_path) + '/tensorboard'
    print(f'Tensorboard : Logdir Tensorboard : {logdir} run with `tensorboard --logdir={logdir}`')
    board = SummaryWriter(log_dir=logdir)
    progress_bar_data = tqdm(iterable=enumerate(dataloader), total=5000, leave=False)
    at = 0
    for epoch in range(train_config.epochs):
        avg = 0
        for index, (input_ids_org, _) in progress_bar_data:
            input_ids_org: torch.Tensor = input_ids_org
            at += 1
            input_ids = make2d(input_ids_org).to(device)
            target_ids = make2d(input_ids_org).to(device)
            model.zero_grad()
            _, loss = model.forward(input_ids=input_ids, target_ids=target_ids)
            avg += loss.item()
            loss.backward()
            if math.isnan(loss):
                exit(0)
            optimizer.step()
            if ((index + 1) % 50) == 0:
                board.add_scalar('TrainLoss', scalar_value=loss.item(), global_step=at)
                board.add_scalar('TrainLossAvg', scalar_value=avg / (index + 1), global_step=at)
            progress_bar_data.set_postfix(loss=loss.item(), avg_loss=avg / (index + 1),
                                          epoch=f'[{epoch} / {train_config.epochs}]')
        save_checkpoints(model=model.state_dict(), optimizer=optimizer.state_dict(),

                         epoch=epoch + 1,
                         name=f'{out_path}/RWKV-model.pt')


if __name__ == '__main__':
    train()
