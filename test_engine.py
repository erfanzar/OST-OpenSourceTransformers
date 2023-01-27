from core.load import load
import torch
from erutils.command_line_interface import fprint

import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--config-path', '-config-path', type=str, default='config/config.yaml', )
pars.add_argument('--model-path', '-model-path', type=str, default='model.pt', )
pars.add_argument('--generate', '-generate', type=int, default=5000, )
opt = pars.parse_args()


def main(opt):
    fprint('Loading the Model ...')
    _, _ = load(path_model=opt.model_path, config_path=opt.config_path,
                generate_token=opt.generate,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                )


if __name__ == "__main__":
    main(opt)
