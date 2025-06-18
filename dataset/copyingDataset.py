import torch
from torch.utils.data import Dataset


class CopyDataset(Dataset):
    def __init__(self, seq_len, vocab_size=100, num_samples=2000):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Генерируем случайные последовательности из [1, vocab_size)
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # таргет = тот же самый тензор

class NoisyCopyDataset(CopyDataset):
    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        noise = torch.zeros(1, dtype=x.dtype)  # токен 0 — шум
        x_noisy = torch.cat([noise, x, noise], dim=0)
        return x_noisy[:self.seq_len], x  # input truncated, target — исход