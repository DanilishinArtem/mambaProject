import torch.nn as nn
import torch.optim as optim
import time
import torch
from dataset.lookUpDataset import LookupDataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./tensorboard/lookup")

def train_and_eval_lookup(model, name, train_loader, test_seq_lens, device, epochs=5):
    global_step = 0
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_of_training = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            global_step += 1
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, L, V)
            # Предполагаем, что модель возвращает распределение по токенам на каждом шаге
            # Ответ должен быть на последнем токене
            last_logits = logits[:, -1, :]  # (B, V)
            loss = criterion(last_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar(f"Loss/{name}", total_loss / global_step, global_step)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss/len(train_loader):.4f}")
    print("Training time:", round(time.time() - start_of_training, 2), "sec")

    # Оценка
    model.eval()
    results = {}
    with torch.no_grad():
        for L in test_seq_lens:
            if L % 2 == 0:
                continue  # только нечётные длины, иначе нельзя вставить query
            test_data = LookupDataset(seq_len=L, vocab_size=train_loader.dataset.vocab_size, num_samples=256)
            x_test = torch.stack([x for x, _ in test_data]).to(device)
            y_test = torch.tensor([y for _, y in test_data]).to(device)
            logits = model(x_test)
            preds = logits[:, -1, :].argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()
            results[L] = acc
            writer.add_scalar(f"Accuracy/{name}", acc, L)
    return results
