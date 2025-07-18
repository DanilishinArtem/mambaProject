import torch
from transformers import AutoTokenizer

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batch_size = 4
    embed_dim = 64
    heads = 4
    num_layers = 12
    num_epochs_wiki = 10
    num_epochs_babilong = 10

# 8k
    max_length = 8192

# # 32k
#     max_length = 32768

# # 64k
#     max_length = 65536

# # 128k
#     max_length = 131072

    mamba = True
    transformer = False