import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from config import Config
import os
class QualityDataset:
    def __init__(self, path_train="/home/adanilishin/mambaProject/dataset/qualityDataset/QuALITY.v1.0.1.train", path_test="/home/adanilishin/mambaProject/dataset/qualityDataset/QuALITY.v1.0.1.dev"):
        self.tokenizer = Config.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = Config.batch_size
        self.max_length = Config.max_length if hasattr(Config, "max_length") else 2048
        self.path_train = path_train
        self.path_test = path_test
        self.train_data = self._prepare_dataset(self.path_train, cache_name="train")
        self.test_data = self._prepare_dataset(self.path_test, cache_name="test")
    
    def tokenize_fn(self, example):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=self.max_length)
    def _load_jsonl(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    def _prepare_dataset(self, path, cache_dir="./mapping", cache_name="train"):
        cached_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cached_path):
            print(f"�� Loading cached dataset from {cached_path}")
            return load_from_disk(cached_path)
        os.makedirs(cache_dir, exist_ok=True)
        raw_data = self._load_jsonl(path)
        formatted_all = []
        for example in raw_data:
            formatted_all.extend(self.format_quality(example))
        hf_dataset = Dataset.from_list(formatted_all)
        tokenized = hf_dataset.map(self.tokenize_fn, batched=False)
        if "text" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["text"])
        tokenized.save_to_disk(cached_path)
        print(f"✅ Saved mapped dataset to {cached_path}")
        return tokenized
    @staticmethod
    def format_quality(example):
        article = example["article"]
        results = []
        for q in example["questions"]:
            if "gold_label" not in q:
                continue  # пропускаем примеры без ответа
            question = q["question"]
            choices = q["options"]
            formatted_choices = " ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            text = f"{article}\nQuestion: {question}\n{formatted_choices}"
            label = int(q["gold_label"])  # предполагается, что это int (0..3)
            results.append({"text": text, "label": label})
        return results
    def collate_fn(self, batch):
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        labels = torch.tensor([x["label"] for x in batch])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {"input_ids": input_ids, "labels": labels}
    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)