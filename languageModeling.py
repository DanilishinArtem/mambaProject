from torch.utils.data import DataLoader
import torch
import pandas as pd
from dataset.languageDataset import WikiTextDataset
from models.mambaPlusPlus import MambaPlusPlusML
from models.transformer import Transformer
from trainer.trainLanguage import train_and_eval_language

import os
os.environ['CURL_CA_BUNDLE'] = ''
# pip install requests==2.27.1

def run_experiment(epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50277
    batch_size = 1

    train_ds = WikiTextDataset(block_size=1024, max_samples=10000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    models = {
        "Transformer": Transformer(vocab_size=vocab_size, embed_dim=1024, nhead=4, num_layers=1)#,
        # "Mamba++": MambaPlusPlusML(vocab_size=vocab_size, embed_dim=1024, hidden_dim=1024, num_heads=4, num_layers=1)
    }

    all_losses = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        train_and_eval_language(
            model=model,
            name=name,
            train_loader=train_loader,
            device=device,
            epochs=epochs
        )

if __name__ == "__main__":
    run_experiment(epochs=3)