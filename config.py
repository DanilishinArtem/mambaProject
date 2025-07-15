import torch
from transformers import AutoTokenizer

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batch_size = 1
    embed_dim = 32
    heads = 2
    num_layers = 2
    num_epochs_wiki = 20
    num_epochs_babilong = 1

# 8k
    max_length = 8192

# # 32k
#     max_length = 32768

# # 64k
#     max_length = 65536

# # 128k
#     max_length = 131072

    mamba = False
    transformer = True