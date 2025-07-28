import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from config import Config
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from mamba_ssm.models.mixer_seq_simple import create_block


def fix_seeds(seed: int = 42):
    # Python
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def give_transformer_block(seq_len):
    from models.alibi import GPTNeoXAlibiLayer
    from transformers import  GPTNeoXConfig
    vocab_size = len(Config.tokenizer)
    config = GPTNeoXConfig(
                bos_token_id=0,
                eos_token_id=0,
                hidden_size=Config.embed_dim,
                intermediate_size=Config.embed_dim*4,
                num_attention_heads=Config.heads,
                num_hidden_layers=Config.num_layers,
                vocab_size=vocab_size,
                max_position_embeddings=seq_len,
                )
    transformer_block = GPTNeoXAlibiLayer(config)
    return transformer_block

def give_vanilla_mamba():
    vanilla_mamba = create_block(
        d_model=Config.embed_dim,
        d_intermediate=0,
        ssm_cfg={"d_state": 32, "layer": "Mamba2"},
        layer_idx=0
    )
    return vanilla_mamba

def give_mamba_block(heads):
    from models.mambaPlusPlus import MambaPlusPlus_layer
    mamba_layer = MambaPlusPlus_layer(Config.embed_dim, heads)
    return mamba_layer

def mode_mamba_block(mamba_block, input):
    return mamba_block(input)[0]




def approximation(transformer_block, mamba_block, seq_len, max_t, tag):
    writer = SummaryWriter(log_dir="./tensorboard/{}".format(tag))
    # === Настройка эксперимента ===
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transformer_block = transformer_block.to(device)
    mamba_block = mamba_block.to(device)

    dim = Config.embed_dim
    batch_size = Config.batch_size

    # freeze transformer
    for p in transformer_block.parameters():
        p.requires_grad = False

    # optimizer = optim.Adam(mamba_block.parameters(), lr=1e-3)
    optimizer = optim.AdamW(mamba_block.parameters(), lr=1e-5, weight_decay=0.1)

    # === Тренировка ===
    for step in range(max_t):
        x = torch.randn(batch_size, seq_len, dim).to(device)

        with torch.no_grad():
            target = transformer_block(x)[0]

        pred = mode_mamba_block(mamba_block=mamba_block, input=x)
        # pred = mamba_block(x)[0]
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss_of_approximation", loss.item(), step)

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.6f}")


if __name__ == "__main__":
    fix_seeds()
    max_t = 50000
    seq_len = 4096
    heads = 1
    transformer_block = give_transformer_block(seq_len)
    # mamba_block = give_mamba_block(heads)
    mamba_block = give_vanilla_mamba()

    print("[INFO] Running approximation procedure ... ")
    approximation(transformer_block=transformer_block, mamba_block=mamba_block, seq_len=seq_len, max_t=max_t, tag="head{}".format(heads))
    print("[INFO] End of approximation ... ")