from torch.utils.data import DataLoader
import torch
import pandas as pd
from dataset.languageDataset import WikiTextDataset
from trainer.trainLanguage import train_and_eval_language
from models.LSTM import LSTM
from models.transformer import Transformer
from models.mamba import Mamba

# from models.mambaPlusPlusSum import MambaPlusPlusML
# from models.mambaPlusPlusConcat import MambaPlusPlusML
from models.mambaPlusPlus import MambaPlusPlusML

def count_parameters(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n[{}] Total: {}, Trainable: {}".format(name, total, trainable))

import os
os.environ['CURL_CA_BUNDLE'] = ''
# pip install requests==2.27.1


def run_experiment(epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50277
    batch_size = 1
    embed_dim = 1024
    heads = 2

    train_ds = WikiTextDataset(block_size=1024, max_samples=1000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    models = {
        # "LSTM": LSTM(vocab_size, embed_dim, hidden_dim),
        "Transformer": Transformer(vocab_size, embed_dim, nhead=heads, num_layers=1),
        # "Mamba": Mamba(vocab_size=vocab_size, hidden_dim=int(hidden_dim / 2), num_layers=1),
        "Mamba++": MambaPlusPlusML(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=embed_dim, num_heads=heads, num_layers=1)
    }

    all_losses = {}
    for name, model in models.items():
        count_parameters(model, name)
        print(f"=== Training {name} ===")
        train_and_eval_language(
            model=model,
            name=name,
            train_loader=train_loader,
            device=device,
            epochs=epochs
        )

if __name__ == "__main__":
    run_experiment(epochs=2)