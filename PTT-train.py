import torch.cuda

from core.train import train_ptt

try:
    from erutils.loggers import fprint
except:
    print('Downloading Missing Module [Erutils]')
    import subprocess
    import sys

    path = sys.executable
    subprocess.run(f'{path} -m pip install erutils')
    from erutils.loggers import fprint
import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--config-path', '-config-path', type=str, default='config/PTTGenerative-small.yaml', )
opt = pars.parse_args()

if __name__ == "__main__":
    fprint(f"Config :: => {opt.config_path}")
    prp = torch.cuda.get_device_properties("cuda")
    fprint(
        f'DEVICES : {torch.cuda.get_device_name()} | {prp.name} |'
        f' {prp.total_memory / 1e9} GB Memory')
    train(config_path=opt.config_path)
