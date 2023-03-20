import argparse
import logging
import pytorch_lightning as pl
import erutils
import torch.utils.data
from erutils.loggers import fprint
from transformers import LlamaTokenizer, AutoTokenizer, PreTrainedTokenizer

from config.config import TQDM_KWARGS
from modules.dataset import DatasetLLMoFC
from modules import LLMoFCForCausalLM, LLMoFCConfig
from utils.utils import make2d, save_checkpoints, get_config_by_name, device_info, get_memory, count_model_parameters, \
    create_output_path, get_data

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--out-path', '--out-path', type=str, default='out')
pars.add_argument('--model', '--model', type=str, default='LLMoFC-ML')
pars.add_argument('--data-src', '--data-src', type=str, default='data/alpaca_data.json')
# HF-kilt_tasks//eli5
options = pars.parse_args()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.CRITICAL)


def main(opt):
    device_info()
    data = get_data(opt.data_src)
    parameters: LLMoFCConfig = get_config_by_name(opt.model)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer_model/BASE')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = DatasetLLMoFC(data=data, max_length=parameters.max_sentence_length, tokenizer=tokenizer)
    parameters.vocab_size = dataset.tokenizer.vocab_size + 1
    parameters.data_path = opt.data_src

    parameters.batch_size = opt.batch
    dataloader = torch.utils.data.DataLoader(dataset=dataset)
    erutils.loggers.show_hyper_parameters(parameters)

    fprint('Loading Model ...' if opt.weight else 'Creating Model ...')

    model = LLMoFCForCausalLM(config=parameters)
    # trainer = pl.Trainer(accelerator="tpu", devices=8, max_epochs=50)
    # trainer = pl.Trainer(accelerator="gpu", max_epochs=50)
    # trainer.fit(model, train_dataloaders=dataloader)
    trainer = pl.Trainer(accelerator="tpu", devices=8, max_epochs=2)

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path='out-a')
    trainer.save_checkpoint('CKPT.pt')


if __name__ == "__main__":
    main(options)
