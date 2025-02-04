import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SequentialDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=50):
        self.X = np.random.randint(0, vocab_size, (num_samples, seq_len))
        self.y = np.random.randint(0, vocab_size, (num_samples,))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def get_dataloader(batch_size=32):
    dataset = SequentialDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)