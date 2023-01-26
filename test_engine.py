from core.load import load
import torch

if __name__ == "__main__":
    load(path='model.pt', path_data='data/input.txt', vocab_size=65,
         generate_token=2000,
         chunk_size=328, n_embedded=324, device='cuda' if torch.cuda.is_available() else 'cpu',
         head_size=64, n_layers=12, n_head=12)
