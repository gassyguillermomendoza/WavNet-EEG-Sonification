import torch
from torch.utils.data import Dataset

class TokenDatasetWithLabels(Dataset):
    def __init__(self, file_path):
        self.tokens, self.labels = torch.load(file_path)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx][: self.max_len - 1]
        y = self.tokens[idx][1 : self.max_len]
        
        inst = self.labels[idx]
        pitch = 60  # e.g. default to middle C

        return x, y, torch.tensor(inst, dtype=torch.long), torch.tensor(pitch, dtype=torch.long)


class NSynthConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_len=16000):
        self.tokens, self.labels = torch.load(path)
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx][: self.max_len - 1]
        y = self.tokens[idx][1 : self.max_len]

        inst, pitch = self.labels[idx]
        return x, y, torch.tensor(inst, dtype=torch.long), torch.tensor(pitch, dtype=torch.long)
