import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter


def train_and_eval_language(model, name, train_loader, device, epochs=3, eval_steps=500):
    writer = SummaryWriter(log_dir="./tensorboard/language")
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    start_time = time.time()

    global_step = 0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (B, L, V)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.item()

            # ⏱ Accuracy измерение
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # (B, L)
                mask = (y != 0)
                correct = ((preds == y) & mask).sum().item()
                total = mask.sum().item()
                total_correct += correct
                total_tokens += total

                acc = correct / total if total > 0 else 0.0
                writer.add_scalar(f"Accuracy/{name}", acc, global_step)

            # ⏱ Logging
            print(f"[{name}] Step {global_step}, Loss = {loss.item():.4f}, Acc = {acc:.4f}")
            writer.add_scalar(f"Loss/{name}", total_loss / global_step, global_step)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        print(f"[{name}] Epoch {epoch+1}/{epochs}, Avg Loss = {avg_loss:.4f}, Avg Acc = {avg_acc:.4f}")

    print("Training time:", round(time.time() - start_time, 2), "sec")