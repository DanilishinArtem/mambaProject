import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from config import Config
from torch.nn.utils.rnn import pad_sequence
import os

class BABILongDataset:
    def __init__(self, dataset_dir="./dataset/babilong", cache_dir="./mapping_babi"):
        self.tokenizer = Config.tokenizer
        self.batch_size = Config.batch_size
        self.max_length = Config.max_length
        self.cache_dir = cache_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data = self._prepare_split(dataset_dir)

    def _prepare_split(self, dataset_dir):
        cache_path = os.path.join(self.cache_dir)
        if os.path.exists(cache_path):
            print(f"🔁 Loading cached dataset from {cache_path}")
            return load_from_disk(cache_path)

        print(f"📦 Loading and tokenizing babilong from {dataset_dir}")
        dataset = load_dataset("parquet", data_files="{}/*parquet".format(dataset_dir))["train"]

        def preprocess(example):
            prompt = f"{example['input']}\n\nQuestion: {example['question']}\nAnswer:"
            target = str(example['target'])

            prompt_enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=False,
            )
            target_enc = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_length - len(prompt_enc['input_ids']),
                return_attention_mask=False,
            )

            input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        dataset = dataset.map(
            preprocess,
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count(),
            load_from_cache_file=False,
        )

        os.makedirs(cache_path, exist_ok=True)
        dataset.save_to_disk(cache_path)
        print(f"✅ Saved processed dataset to {cache_path}")
        return dataset

    def collate_fn(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]

        input_ids = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in attention_mask],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in labels],
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_data_loader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True
        )