import numpy as np
import torch

class LlamaDataset(torch.utils.data.Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y
