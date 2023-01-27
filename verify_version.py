import subprocess

packages = [
    'pip3 install --upgrade --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116',
    'pip3 install --upgrade erutils'
]

if __name__ == "__main__":
    for package in packages:
        subprocess.run(package)
