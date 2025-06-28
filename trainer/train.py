import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import load as load_metric
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from config import Config


def train_model(model, dataloader, writer, tag, epochs=1):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    step = 0
    total_loss = 0

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Training {tag} Epoch {epoch+1}"):
            step += 1
            input_ids = batch["input_ids"].to(Config.device)      # shape: (batch, seq_len)
            labels = batch["labels"].to(Config.device)            # shape: (batch)

            optimizer.zero_grad()
            logits = model(input_ids)                              # shape: (batch, seq_len, vocab_size)
            logits_last = logits[:, -1, :]                         # берем логиты для последнего токена
            loss = loss_fn(logits_last, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar("Train/Loss", total_loss / step, step)

        print(f"[{tag}] Epoch {epoch+1} - Avg Loss: {total_loss / step:.4f}")


def evaluate_model(model, dataloader, tag):
    metric = load_metric("accuracy")
    model.eval()
    print(f"Test loader length: {len(dataloader)}")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {tag}"):
            input_ids = batch["input_ids"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            logits = model(input_ids)
            preds = logits[:, -1, :].argmax(dim=-1)
            metric.add_batch(predictions=preds.cpu().numpy(), references=labels.cpu().numpy())

    return metric.compute()