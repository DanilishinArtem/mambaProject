from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from dataset.copyDataset import CopyDataset, NoisyCopyDataset
from models.LSTMCopy import LSTMCopy
from models.transformerCopy import TransformerCopy
from models.mambaPlusPlus import MambaPlusPlus
from trainer.train_eval import train_and_eval


def generate_synthetic_data(batch_size, seq_len, vocab_size, top_k=1):
    # Последовательности: случайные токены от 1 до vocab_size-1 (0 — паддинг)
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    # Значения токенов можно взять как индексы (или любые веса)
    values = x.float()
    
    # Для задачи 1: метка — позиция токена с максимальным значением
    if top_k == 1:
        max_vals, max_indices = values.max(dim=1)
        labels = torch.zeros_like(x)
        labels.scatter_(1, max_indices.unsqueeze(1), 1)  # one-hot маска
    
    # Для задачи 2: топ-k
    else:
        topk_vals, topk_indices = torch.topk(values, top_k, dim=1)
        labels = torch.zeros_like(x)
        labels.scatter_(1, topk_indices, 1)
    
    return x, labels


import torch.nn as nn
import torch.optim as optim

# Тренировка
def train_epoch(model, criterion, optimizer, batch_size=64, seq_len=20, top_k=1):
    model.train()
    x, labels = generate_synthetic_data(batch_size, seq_len, vocab_size, top_k)
    optimizer.zero_grad()
    logits = model(x)  # (B, L, vocab_size)
    # Для сравнения нужно получить "важность" токенов, возьмем max логит по vocab_size
    # Предполагаем, что модель должна выделить важные токены, поэтому берем max по vocab_size
    logits_max, _ = logits.max(dim=2)  # (B, L)
    loss = criterion(logits_max, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    num_heads = 4

    model = MambaPlusPlus(vocab_size, embed_dim, hidden_dim, num_heads)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_epoch(model, criterion, optimizer)
