from torch.utils.data import DataLoader
import torch
import pandas as pd
from dataset.lookUpDataset import LookupDataset
from trainer.trainLookUp import train_and_eval_lookup
from models.LSTM import LSTM
from models.transformer import Transformer
from models.mamba import Mamba

# from models.mambaPlusPlusSum import MambaPlusPlusML
from models.mambaPlusPlusConcat import MambaPlusPlusML
# from models.mambaPlusPlus import MambaPlusPlusML

def count_parameters(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n[{}] Total: {}, Trainable: {}".format(name, total, trainable))


def run_experiment(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100
    embed_dim = 32
    seq_len = 21  # 10 пар + 1 запрос
    batch_size = 64
    num_samples = 10000
    epochs = 5
    heads = 2

    train_ds = LookupDataset(seq_len=seq_len, vocab_size=vocab_size, num_samples=num_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_seq_lens = [11, 21, 41]

    models = {
        # "LSTM": LSTM(vocab_size, embed_dim, hidden_dim),
        "Transformer": Transformer(vocab_size, embed_dim, nhead=heads, num_layers=1),
        # "Mamba": Mamba(vocab_size=vocab_size, hidden_dim=int(hidden_dim / 2), num_layers=1),
        "Mamba++": MambaPlusPlusML(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=embed_dim, num_heads=heads, num_layers=1)
    }

    all_results = {}
    for name, model in models.items():
        count_parameters(model, name)
        print(f"=== Training {name} ===")
        results = train_and_eval_lookup(
            model=model,
            name=name,
            train_loader=train_loader,
            test_seq_lens=test_seq_lens,
            device=device,
            epochs=epochs
        )
        all_results[name] = results

    # Вывод таблицы
    df = pd.DataFrame(all_results).T
    df.columns = [f"len={L}" for L in df.columns]
    print("=== Final Accuracy ===")
    print(df)

if __name__ == "__main__":
    run_experiment(5)
