import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from config import Config
import math


def train_model(model, dataloader, writer, tag, epochs=1):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(dataloader, desc=f"Training {tag} Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(Config.device)        # (B, L)
            labels = batch["labels"].to(Config.device)              # (B, L)

            optimizer.zero_grad()
            logits = model(input_ids)                               # (B, L, V)

            loss = loss_fn(logits['logits'].view(-1, logits['logits'].size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                active_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * active_tokens
                total_tokens += active_tokens

            step += 1
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Perplexity", math.exp(loss.item()), step)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(f"[{tag}] Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")


def evaluate_model(model, dataloader, tag):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {tag}"):
            input_ids = batch["input_ids"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            active_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"[{tag}] Evaluation - Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")