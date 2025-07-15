import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk
from config import Config


class WikiTextDataset:
    def __init__(self, dataset_dir="./dataset/babilong", cache_dir="./mapping_wiki"):
        self.tokenizer = Config.tokenizer
        self.batch_size = Config.batch_size
        self.cache_dir = cache_dir
        self.chunk_size = Config.max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = self._prepare_dataset(dataset_dir)

    def _prepare_dataset(self, dataset_dir):
        if os.path.exists(self.cache_dir):
            print(f"🔁 Loading cached WikiText split from {self.cache_dir}")
            return load_from_disk(self.cache_dir)

        print(f"📦 Loading and tokenizing WikiText from: {dataset_dir}")
        raw_dataset = load_dataset("parquet", data_files="{}/*parquet".format(dataset_dir))["train"]

        def tokenize(example):
            return self.tokenizer(example["text"], return_attention_mask=False)

        tokenized = raw_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
            num_proc=os.cpu_count(),
            load_from_cache_file=False
        )

        # Разбиваем в непрерывные чанки фиксированной длины
        def group_texts(examples):
            concatenated = sum(examples["input_ids"], [])
            total_length = (len(concatenated) // self.chunk_size) * self.chunk_size
            chunks = [concatenated[i:i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            return {"input_ids": chunks}

        grouped = tokenized.map(
            group_texts,
            batched=True,
            num_proc=os.cpu_count()
        )

        def create_labels(example):
            return {
                "attention_mask": [1] * len(example["input_ids"]),
                "labels": example["input_ids"].copy()
            }

        final_dataset = grouped.map(create_labels)

        os.makedirs(self.cache_dir, exist_ok=True)
        final_dataset.save_to_disk(self.cache_dir)
        print(f"✅ Saved tokenized WikiText to {self.cache_dir}")
        return final_dataset

    def collate_fn(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_data_loader(self, shuffle=True):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=True
        )