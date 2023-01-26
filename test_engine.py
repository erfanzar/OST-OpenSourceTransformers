from core.load import load
import torch

if __name__ == "__main__":
    load(path='model.pt', path_data='data/input.txt', vocab_size=65,
         generate_token=2000,
         chunk_size=728, number_of_embedded=324, device='cuda' if torch.cuda.is_available() else 'cpu',
         head_size=64, number_of_layers=24, number_of_head=12)