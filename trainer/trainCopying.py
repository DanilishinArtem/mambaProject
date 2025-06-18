import torch.optim as optim
import torch.nn as nn
import torch
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./tensorboard/copying")

def train_and_eval(model, name, train_loader, test_seq_lens, device, epochs=5):
    global_step = 0
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Тренировка
    model.train()
    start_of_training = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            global_step += 1
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, L, V)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar("Loss/{}".format(name), total_loss / global_step, global_step)
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(train_loader):.4f}")
    end_of_training = time.time()
    print("Time of training: {}".format(round(end_of_training - start_of_training, 4)))
    # Оценка точности на разных длинах
    model.eval()
    results = {}
    start_of_inference = time.time()
    with torch.no_grad():
        global_step = 0
        for L in test_seq_lens:
            global_step += 1
            # Формируем батч фиксированного размера
            x_test = torch.randint(1, train_loader.dataset.vocab_size, (64, L), device=device)
            y_test = x_test
            logits = model(x_test)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()
            results[L] = acc
            writer.add_scalar("Accuracy/{}".format(name), acc, L)
    end_of_inference = time.time()
    print("Time of inference: {}".format(round(end_of_inference - start_of_inference, 4)))
    return results