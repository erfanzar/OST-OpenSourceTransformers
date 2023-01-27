from core.train import train

try:
    from erutils.command_line_interface import fprint
except:
    print('Downloading Missing Module [Erutils]')
    import subprocess
    import sys

    path = sys.executable
    subprocess.run(f'{path} -m pip install erutils')
    from erutils.command_line_interface import fprint
import argparse

pars = argparse.ArgumentParser()
pars.add_argument('--config-path', '-config-path', type=str, default='config/config.yaml', )
opt = pars.parse_args()

if __name__ == "__main__":
    fprint(f"Config :: => {opt.config_path}")
    train(config_path=opt.config_path)
