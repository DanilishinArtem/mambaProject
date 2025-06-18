from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class WikiTextDataset(Dataset):
    def __init__(self, block_size=1024, max_samples=None):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        if max_samples:
            dataset = dataset.select(range(max_samples))

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], return_attention_mask=False)

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        all_input_ids = sum(tokenized["input_ids"], [])
        total_length = (len(all_input_ids) // block_size) * block_size
        all_input_ids = all_input_ids[:total_length]

        self.examples = [
            torch.tensor(all_input_ids[i : i + block_size], dtype=torch.long)
            for i in range(0, total_length, block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx][:-1]  # input_ids, без последнего токена
        y = self.examples[idx][1:]   # labels — сдвинуты на 1 вправо
        return x, y
