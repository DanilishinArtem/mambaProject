import os
import torch
import pandas as pd
from trainer.train import train_model, evaluate_model
from models.mambaPlusPlus import MambaPlusPlusML
from models.transformer import Transformer
from config import Config
from dataset.Babilong import BABILongDataset
from dataset.Wikitext import WikiTextDataset
from torch.utils.tensorboard import SummaryWriter


def print_model_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Total parameters: {total_params:,}")


def create_mamba_model():
    # from mamba_ssm.models.config_mamba import MambaConfig
    # from models.mamba import MambaLMHeadModel
    # vocab_size = len(Config.tokenizer)
    # print("[INFO] vocab_size = {}".format(vocab_size))
    # config = MambaConfig(
    #     d_model=Config.embed_dim,
    #     n_layer=Config.num_layers,
    #     ssm_cfg={"d_state": 16},
    #     vocab_size=vocab_size,
    #     # tie_embeddings=False,
    #     # residual_in_fp32=False
    # )
    # mamba = MambaLMHeadModel(config).to(Config.device)
    # return mamba 

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
    from transformers import  GPTNeoXConfig
    from models.alibi import GPTNeoXAlibiForCausalLM
    vocab_size = len(Config.tokenizer)
    print("[INFO] vocab_size = {}".format(vocab_size))
    config = GPTNeoXConfig(
                bos_token_id=0,
                eos_token_id=0,
                hidden_size=Config.embed_dim,
                intermediate_size=Config.embed_dim*4,
                num_attention_heads=Config.heads,
                num_hidden_layers=Config.num_layers,
                vocab_size=vocab_size,
                max_position_embeddings=1024,
                )
    transformer = GPTNeoXAlibiForCausalLM(config)
    transformer = transformer.to(Config.device)
    return transformer

    # vocab_size = len(Config.tokenizer)
    # print("[INFO] vocab_size = {}".format(vocab_size))
    # transformer = Transformer(
    #     vocab_size,
    #     Config.embed_dim,
    #     nhead=Config.heads,
    #     num_layers=Config.num_layers,
    #     max_seq_len=Config.max_length,
    # ).to(Config.device)
    # return transformer


def train_process(train_loader, tag, num_epochs):
    print("Training models ...")

    if Config.mamba:
        mamba_model = create_mamba_model()
        print_model_parameters(mamba_model, "mamba_model")
        os.makedirs("./checkpoints/{}/mambaPlusPlus".format(tag), exist_ok=True)
        writer_mamba = SummaryWriter(log_dir="./tensorboard/{}/MambaPlusPlus".format(tag))
        print("[INFO] Training MambaPlusPlus...")
        train_model(mamba_model, train_loader, writer_mamba, "Mamba++", epochs=num_epochs)
        torch.save(mamba_model.state_dict(), "./checkpoints/{}/mambaPlusPlus/mamba.pt".format(tag))
    
    if Config.transformer:
        transformer_model = create_transformer_model()
        print_model_parameters(transformer_model, "transformer_model")
        os.makedirs("./checkpoints/{}/transformer".format(tag), exist_ok=True)
        writer_transformer = SummaryWriter(log_dir="./tensorboard/{}/Transformer".format(tag))
        print("[INFO] Training Transformer...")
        train_model(transformer_model, train_loader, writer_transformer, "Transformer", epochs=num_epochs)
        torch.save(transformer_model.state_dict(), "./checkpoints/{}/transformer/transformer.pt".format(tag))


def evaluate_process(eval_loader, tag):
    print("Evaluating models on {}".format(tag))

    if Config.mamba:
        mamba_model = create_mamba_model()
        mamba_model.load_state_dict(torch.load("./checkpoints/{}/mambaPlusPlus/mamba.pt".format(tag)))
        print("Evaluating MambaPlusPlus...")
        evaluate_model(mamba_model, eval_loader, "MambaPlusPlus")

    if Config.transformer:
        transformer_model = create_transformer_model()
        transformer_model.load_state_dict(torch.load("./checkpoints/{}/transformer/transformer.pt".format(tag)))
        print("Evaluating Transformer...")
        evaluate_model(transformer_model, eval_loader, "Transformer")


def finetune_process(train_loader, tag, num_epochs):
    print("Finetuning models ...")

    if Config.mamba:
        mamba_model = create_mamba_model()
        mamba_model.load_state_dict(torch.load("./checkpoints/Wikitext/mambaPlusPlus/mamba.pt".format(tag)))
        writer_mamba = SummaryWriter(log_dir="./tensorboard/{}/MambaPlusPlus".format(tag))
        print("[INFO] Finetuning MambaPlusPlus...")
        train_model(mamba_model, train_loader, writer_mamba, "Mamba++", epochs=Config.num_epochs)

    if Config.transformer:
        transformer_model = create_transformer_model()
        transformer_model.load_state_dict(torch.load("./checkpoints/Wikitext/transformer/transformer.pt".format(tag)))
        writer_transformer = SummaryWriter(log_dir="./tensorboard/{}/Transformer".format(tag))
        print("[INFO] Finetuning Transformer...")
        train_model(transformer_model, train_loader, writer_transformer, "Transformer", epochs=num_epochs)



if __name__ == "__main__":
    # train and evaluation models for dataset Wikitext ...
    data = WikiTextDataset(dataset_dir="./dataset/wikitext")
    dataLoader = data.get_data_loader()
    train_process(dataLoader, "Wikitext", Config.num_epochs_wiki)
    evaluate_process(dataLoader, "Wikitext")

    # # train and evaluation models for dataset Babilon ...
    # if Config.max_length == 8192:
    #     path = "./dataset/babilong/8k"
    # elif Config.max_length == 32768:
    #     path = "./dataset/babilong/32k"
    # elif Config.max_length == 65536:
    #     path = "./dataset/babilong/64k"
    # else:
    #     path = "./dataset/babilong/128k"

    # data = BABILongDataset(dataset_dir=path)
    # dataLoader = data.get_data_loader()
    # # train_process(dataLoader, "Babilong", Config.num_epochs_babilong)
    # finetune_process(dataLoader, "Babilong", Config.num_epochs_babilong)
    # evaluate_process(dataLoader, "Babilong")