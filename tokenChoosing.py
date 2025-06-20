from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from dataset.tokenChoosingDataset import TopKTokenSelectionDataset
from trainer.trainTokenChoosing import train_and_eval_topk
from models.mambaPlusPlus import MambaPlusPlus_regular
from models.LSTM import LSTM
from models.transformer import Transformer

def run_experiment(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    num_heads = 4
    seq_len = 32
    batch_size = 64
    num_samples = 10000
    top_k = 3
    epochs = 5

    # Инициализация данных и модели
    train_ds = TopKTokenSelectionDataset(seq_len=seq_len, vocab_size=vocab_size, num_samples=num_samples, top_k=top_k)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_seq_lens = [16, 32, 64]

    models = {
        "LSTM": LSTM(vocab_size, embed_dim, hidden_dim),
        "Transformer": Transformer(vocab_size, embed_dim, nhead=2, num_layers=8),
        "Mamba++": MambaPlusPlus_regular(vocab_size, embed_dim, hidden_dim, num_heads=2)
    }

    all_results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        results = train_and_eval_topk(
            model=model,
            name=name,
            train_loader=train_loader,
            test_seq_lens=test_seq_lens,
            device=device,
            epochs=epochs,
            top_k=top_k
        )
        all_results[name] = results

    # Вывод таблицы
    df = pd.DataFrame(all_results).T
    df.columns = [f"len={L}" for L in df.columns]
    print("\n=== Final Accuracy ===")
    print(df)

if __name__ == "__main__":
    run_experiment(5)