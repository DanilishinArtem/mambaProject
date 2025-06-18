from torch.utils.data import Dataset
import torch

class TopKTokenSelectionDataset(Dataset):
    def __init__(self, seq_len, vocab_size=100, num_samples=2000, top_k=1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.top_k = top_k

        # Случайные последовательности из [1, vocab_size)
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))

        # Генерация меток для задач топ-K
        self.labels = torch.zeros_like(self.data)
        values = self.data.float()
        if self.top_k == 1:
            max_vals, max_indices = values.max(dim=1)
            self.labels.scatter_(1, max_indices.unsqueeze(1), 1)
        else:
            topk_vals, topk_indices = torch.topk(values, self.top_k, dim=1)
            self.labels.scatter_(1, topk_indices, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
