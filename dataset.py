import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, token_file):
        self.token_seqs = torch.load(token_file)

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        tokens = self.token_seqs[idx]
        return tokens[:-1], tokens[1:]  # input, target
