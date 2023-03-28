import os
import sys

sys.path.append('/'.join(os.getcwd().split('\\')[:-1]))
import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--model-type', '--model-type', type=str, default='LGeM-X')
pars.add_argument('--tokenizer', '--tokenizer', type=str, default='tokenizer_model/BASE')
opt = pars.parse_args()

from modules import LGeMForCausalLM
import accelerate
from utils.utils import get_config_by_name, count_model_parameters,prompt_to_instruction


def _main(options):
    accelerator = accelerate.Accelerator()
    config = get_config_by_name(opt.model_type)

    config.vocab_size = 32000
    with accelerate.init_empty_weights():
        model = LGeMForCausalLM(config).half()
    print(count_model_parameters(model))


if __name__ == "__main__":
    _main(opt)
