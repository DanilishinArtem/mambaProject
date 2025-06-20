import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./tensorboard/token_selection")

def train_and_eval_topk(model, name, train_loader, test_seq_lens, device, epochs=5, top_k=1):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Тренировка
    model.train()
    start_of_training = time.time()
    global_step = 0
    total_loss = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            global_step += 1
            x, y = x.to(device), y.to(device).float()
            logits = model(x)  # (B, L, V)
            logits_max, _ = logits.max(dim=2)  # (B, L) — логиты по наибольшему классу на каждом токене

            loss = criterion(logits_max, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar(f"Loss/{name}", total_loss / global_step, global_step)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss / len(train_loader):.4f}")
    end_of_training = time.time()
    print(f"Time of training: {round(end_of_training - start_of_training, 2)} sec")

    # Оценка
    model.eval()
    results = {}
    start_of_inference = time.time()
    with torch.no_grad():
        for L in test_seq_lens:
            x_test = torch.randint(1, train_loader.dataset.vocab_size, (64, L), device=device)
            
            # Генерация правильных топ-K меток
            labels = torch.zeros_like(x_test)
            topk_vals, topk_indices = torch.topk(x_test.float(), top_k, dim=1)
            labels.scatter_(1, topk_indices, 1)

            logits = model(x_test)  # (64, L, V)
            logits_max, _ = logits.max(dim=2)  # (64, L)
            preds = (torch.sigmoid(logits_max) > 0.5).float()

            correct = (preds == labels).float().mean().item()
            results[L] = correct
            writer.add_scalar(f"Accuracy/{name}", correct, L)
            print(f"Eval seq_len={L}: Accuracy={correct:.4f}")
    end_of_inference = time.time()
    print(f"Time of inference: {round(end_of_inference - start_of_inference, 2)} sec")
    return results
