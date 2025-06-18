from torch.utils.data import Dataset
import torch

class LookupDataset(Dataset):
    def __init__(self, seq_len, vocab_size=100, num_samples=2000):
        super().__init__()
        assert seq_len % 2 == 1, "seq_len must be odd (context pairs + 1 query)"
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.num_pairs = (seq_len - 1) // 2  # (key, value) pairs

        self.data = []
        for _ in range(num_samples):
            # Генерируем уникальные ключи и значения
            keys = torch.randint(1, vocab_size, (self.num_pairs,))
            values = torch.randint(1, vocab_size, (self.num_pairs,))
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            query_key = keys[query_idx]
            target_value = values[query_idx]

            # Собираем последовательность: [k1,v1,k2,v2,...,query_key]
            sequence = torch.empty(self.seq_len, dtype=torch.long)
            sequence[:-1:2] = keys
            sequence[1:-1:2] = values
            sequence[-1] = query_key

            self.data.append((sequence, target_value))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
