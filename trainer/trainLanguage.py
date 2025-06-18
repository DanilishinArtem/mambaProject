import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./tensorboard/pile")

def train_and_eval_pile(model, name, train_loader, device, epochs=3, eval_steps=500):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Для PAD токенов, если применимо

    global_step = 0
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            global_step += 1
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B, L, V)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar(f"Loss/{name}", total_loss / global_step, global_step)

            if global_step % eval_steps == 0:
                print(f"[{name}] Step {global_step}, Loss = {loss.item():.4f}")

        print(f"[{name}] Epoch {epoch+1}/{epochs}, Avg Loss = {total_loss/len(train_loader):.4f}")

    print("Training time:", round(time.time() - start_time, 2), "sec")
