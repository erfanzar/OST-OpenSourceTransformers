import argparse
import os

TPU = False
COLAB = True
USE_JIT = '1'
os.environ['USE_JIT'] = USE_JIT
from transformers import LlamaForCausalLM, LlamaConfig
if TPU:
    if COLAB:
        import requests
        import os

        if 'TPU_DRIVER_MODE' not in globals():
            url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver_nightly'
            resp = requests.post(url)
            TPU_DRIVER_MODE = 1

        # TPU driver as backend for JAX
        from jax.config import config

        config.FLAGS.jax_xla_backend = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
        print(config.FLAGS.jax_backend_target)
        os.environ['JAX_PLATFORMS'] = ''
    else:
        raise NotImplementedError
else:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader
from modules.datasets import CasualLMDataset
import jax
from utils.utils import get_data
from modules import LGemModelForCasualLM, LGemConfig
from core.jax_trainer import train

pars = argparse.ArgumentParser()

pars.add_argument('--batch', '--batch', type=int, default=1)
pars.add_argument('--train', '--train', type=bool, default=True)
pars.add_argument('--compile', '--compile', type=bool, default=True)
pars.add_argument('--weight', '--weight', type=str, default=None)
pars.add_argument('--accumulate', '--accumulate', type=int, default=4)
pars.add_argument('--out-path', '--out-path', type=str, default='out')
pars.add_argument('--model', '--model', type=str, default='LGeM-S')
pars.add_argument('--save-on-step', '--save-on-step', type=int, default=5000)
pars.add_argument('--data-src', '--data-src', type=str, default='data/alpaca_data.json')
# HF-kilt_tasks//eli5
options = pars.parse_args()


#
# logger = logging.getLogger(__name__)
#
# logging.basicConfig(level=logging.WARN)


def main(opt):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer_model/BASE')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    data = get_data(opt.data_src)[:5000]
    conf: LGemConfig = LGemConfig(
        hidden_size=512,
        intermediate_size=768 * 3,
        num_hidden_layers=12,
        num_attention_heads=8,
        vocab_size=32000,
        dtype=jax.numpy.float32
    )
    model = LGemModelForCasualLM(conf)
    # Replace with your own Dataset
    dataloader = DataLoader(
        CasualLMDataset(data=data, max_length=conf.max_sequence_length, tokenizer=tokenizer, return_tensors='np'),
        batch_size=1, shuffle=True)
    print(model)
    train(model=model, config=conf,
          data_loader=dataloader, total=5000)


if __name__ == "__main__":
    main(options)
