import os
import sys

sys.path.append('/'.join(os.getcwd().split('\\')[:-1]))
import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--model-type', '--model-type', type=str, default='LGeM-ML')
pars.add_argument('--agent-name', '--agent-name', type=str, default='<LLmP> :')
pars.add_argument('--tokenizer', '--tokenizer', type=str, default='tokenizer_model/BASE')
opt = pars.parse_args()

from modules import LGeMForCausalLM, LGeMConfig
import accelerate
from utils.utils import get_config_by_name, count_model_parameters
from transformers import AutoModel


def _main(options):
    accelerator = accelerate.Accelerator()
    config = get_config_by_name(opt.model_type)
    native_config = LGeMConfig(

    )
    config.vocab_size = 32000
    with accelerate.init_empty_weights():
        model = LGeMForCausalLM(config)

    print(count_model_parameters(model))


if __name__ == "__main__":
    _main(opt)
