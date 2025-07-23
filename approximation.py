import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from config import Config
from torch.utils.tensorboard import SummaryWriter

def give_transformer_block():
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
                max_position_embeddings=1024,
                )
    transformer_block = GPTNeoXAlibiLayer(config)
    return transformer_block

def give_mamba_block():
    heads = 32
    from models.mambaPlusPlus import MambaPlusPlus_layer
    mamba_layer = MambaPlusPlus_layer(Config.embed_dim, heads)
    return mamba_layer



def mode_mamba_block(mamba_block, input):
    return mamba_block(input)[0]




def approximation(transformer_block, mamba_block, max_t, tag):
    writer = SummaryWriter(log_dir="./tensorboard/{}".format(tag))
    # === Настройка эксперимента ===
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transformer_block = transformer_block.to(device)
    mamba_block = mamba_block.to(device)

    dim = Config.embed_dim
    seq_len = 32
    batch_size = Config.batch_size

    # freeze transformer
    for p in transformer_block.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(mamba_block.parameters(), lr=1e-3)

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
    max_t = 30000
    transformer_block = give_transformer_block()
    mamba_block = give_mamba_block()

    print("[INFO] Running approximation procedure ... ")
    approximation(transformer_block=transformer_block, mamba_block=mamba_block, max_t=max_t, tag="bugged")
    print("[INFO] End of approximation ... ")