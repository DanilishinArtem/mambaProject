from torch.utils.data import DataLoader
import torch
import pandas as pd
from dataset.languageDataset import WikiTextDataset
from models.mambaPlusPlus import MambaPlusPlusML
from models.transformer import TransformerML
from trainer.trainLanguage import train_and_eval_pile

def run_experiment(epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50277  # Для GPT-NeoX tokenizer
    block_size = 1024
    batch_size = 4
    max_samples = 5000  # для ускоренного теста

    train_ds = WikiTextDataset(block_size=1024, max_samples=10000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    models = {
        "Transformer": TransformerML(vocab_size=vocab_size, embed_dim=2048, num_heads=16, num_layers=24),
        "Mamba++": MambaPlusPlusML(vocab_size=vocab_size, embed_dim=2048, hidden_dim=4096, num_layers=24)
    }

    all_losses = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        train_and_eval_pile(
            model=model,
            name=name,
            train_loader=train_loader,
            device=device,
            epochs=epochs
        )
        # Здесь можно добавить замер perplexity и throughput при необходимости

if __name__ == "__main__":
    run_experiment(epochs=3)
