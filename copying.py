from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from dataset.copyDataset import CopyDataset, NoisyCopyDataset
from models.LSTMCopy import LSTMCopy
from models.transformerCopy import TransformerCopy
from models.mambaPlusPlus import MambaPlusPlus
from trainer.train_eval import train_and_eval

def run_experiment(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100
    embed_dim = 8
    hidden_dim = 16

    # Датасет длины 50
    train_ds = CopyDataset(seq_len=50, vocab_size=vocab_size, num_samples=2000)
    # train_ds = NoisyCopyDataset(seq_len=50, vocab_size=vocab_size, num_samples=2000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    test_seq_lens = [50, 100, 200, 300, 500, 1000]

    models = {
        "LSTM": LSTMCopy(vocab_size, embed_dim, hidden_dim),
        "Transformer": TransformerCopy(vocab_size, embed_dim, nhead=2, num_layers=8),
        "Mamba++": MambaPlusPlus(vocab_size, embed_dim, hidden_dim, num_heads=2)
    }

    all_results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        results = train_and_eval(model, name, train_loader, test_seq_lens, device, epochs=epochs)
        all_results[name] = results

    # Вывод таблицы
    df = pd.DataFrame(all_results).T
    df.columns = [f"len={L}" for L in df.columns]
    print("\n=== Final Accuracy ===")
    print(df)

if __name__ == "__main__":
    run_experiment(5)