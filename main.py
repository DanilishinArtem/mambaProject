import os
import pandas as pd
from dataset.QuALITY import QualityDataset
from trainer.qualityTrain import train_model, evaluate_model
from torch.utils.tensorboard import SummaryWriter
from models.mambaPlusPlus import MambaPlusPlusML
from models.transformer import Transformer
from config import Config
import torch

def print_model_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Total parameters: {total_params:,}")

def create_mamba_model():
    vocab_size = len(Config.tokenizer)
    print("[INFO] vocab_size = {}".format(vocab_size))
    mamba = MambaPlusPlusML(
        vocab_size=vocab_size,
        dim=Config.embed_dim,
        num_heads=Config.heads,
        num_layers=Config.num_layers,
        max_seq_len=Config.max_length,
    ).to(Config.device)
    return mamba

def create_transformer_model():
    vocab_size = len(Config.tokenizer)
    print("[INFO] vocab_size = {}".format(vocab_size))
    transformer = Transformer(
        vocab_size,
        Config.embed_dim,
        nhead=1,
        num_layers=Config.num_layers,
        max_seq_len=Config.max_length,
    ).to(Config.device)
    return transformer


def train_quality(datasets):
    print("Training models on QuALITY dataset...")
    train_loader = datasets.get_train_loader()

    mamba_model = create_mamba_model()
    # transformer_model = create_transformer_model()

    print_model_parameters(mamba_model, "mamba_model")
    # print_model_parameters(transformer_model, "transformer_model")

    os.makedirs("./checkpoints/quality/mambaPlusPlus", exist_ok=True)
    os.makedirs("./checkpoints/quality/transformer", exist_ok=True)

    writer_mamba = SummaryWriter(log_dir="./tensorboard/quality/MambaPlusPlus")
    print("�� Training MambaPlusPlus...")
    train_model(mamba_model, train_loader, writer_mamba, "Mamba++", epochs=Config.num_epochs)
    torch.save(mamba_model.state_dict(), "./checkpoints/quality/mambaPlusPlus/mamba.pt")

    # writer_transformer = SummaryWriter(log_dir="./tensorboard/quality/Transformer")
    # print("�� Training Transformer...")
    # train_model(transformer_model, train_loader, writer_transformer, "Transformer", epochs=Config.num_epochs)
    # torch.save(transformer_model.state_dict(), "./checkpoints/quality/transformer/transformer.pt")


def evaluate_quality(datasets):
    print("Evaluating models on QuALITY dataset...")
    test_loader = datasets.get_test_loader()
    mamba_model, transformer_model = create_models()

    mamba_model.load_state_dict(torch.load("./checkpoints/quality/mambaPlusPlus/mamba.pt"))
    transformer_model.load_state_dict(torch.load("./checkpoints/quality/transformer/transformer.pt"))

    print("Evaluating MambaPlusPlus...")
    acc_mamba = evaluate_model(mamba_model, test_loader, "MambaPlusPlus")
    print(f"✅ MambaPlusPlus Accuracy: {acc_mamba['accuracy']:.4f}")

    print("Evaluating Transformer...")
    acc_transformer = evaluate_model(transformer_model, test_loader, "Transformer")
    print(f"✅ Transformer Accuracy: {acc_transformer['accuracy']:.4f}")


if __name__ == "__main__":
    datasets = QualityDataset()  # по умолчанию пути к train/test в конструкторе
    train_quality(datasets)
    # evaluate_quality(datasets)