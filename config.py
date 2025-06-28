import torch
from transformers import AutoTokenizer

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batch_size = 1
    max_length = 2048
    embed_dim = 128
    heads = 2
    num_layers = 2