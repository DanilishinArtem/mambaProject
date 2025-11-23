import torch
from lib.mamba import create_block
from mamba_ssm.modules.mamba2 import Mamba2
import random
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import functional as F
import os
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

def fix_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def save_mamba_layer(mamba_layer, idx: int):
    save_dir = f'./mamba_layer/model/layer{idx}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        mamba_layer.state_dict(),
        f"{save_dir}/pytorch_model.bin"
    )
    config = {
        "hidden_size": mamba_layer.mixer.d_model,
        "d_state": 32,
        "layer_type": "Mamba2"
    }
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

def get_mamba_layer(layer_path: str):
    config = AutoConfig.from_pretrained(layer_path)
    layer = create_block(
        d_model=config.hidden_size,
        d_intermediate=0,
        ssm_cfg={"d_state": 32, "layer": "Mamba2"},
        layer_idx=0
    )
    print(f'[INFO] Mamba layer: {layer}')
    return layer

def get_transformer_layer(layer_path: str):
    config = AutoConfig.from_pretrained(layer_path)
    layer = OPTDecoderLayer(config)
    state_dict = torch.load(f"{layer_path}/pytorch_model.bin", map_location="cpu")
    layer.load_state_dict(state_dict)
    print(f'[INFO] Transformer layer: {layer}')
    return layer

def get_emb_tokens():
    name = 'facebook/opt-125m'
    config = AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_config(config)
    model_name = './transformer_layer/model/'
    state_dict = torch.load(f'{model_name}/full_model/pytorch_model.bin')
    model.load_state_dict(state_dict)
    return model.model.decoder.embed_tokens.to(torch.float32).cuda()

def training(mamba_layer, transformer_layer, number_of_layer: int, layer_path: str, max_t: int = 5000):
    config = AutoConfig.from_pretrained(layer_path)
    dim = config.hidden_size
    batch_size = 4
    seq_len = 2048
    using_dataset = True

    if using_dataset:
        dataset = load_dataset("hazyresearch/based-fda", cache_dir="~/.cache/huggingface/datasets/")
        texts = dataset["validation"]["text"][:max_t]
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )
        tensor_dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
        batches = list(dataloader)
        MAX_LEN = len(batches)
        emb_tokens = get_emb_tokens()

    writer = SummaryWriter(log_dir="./tensorboard/layer_{}".format(number_of_layer))
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transformer_layer = transformer_layer.to(device)
    mamba_layer = mamba_layer.to(device)
    for p in transformer_layer.parameters():
        p.requires_grad = False
    # optimizer = optim.Adam(mamba_block.parameters(), lr=1e-3)
    optimizer = optim.AdamW(mamba_layer.parameters(), lr=1e-5, weight_decay=0.1)
    
    for step in range(max_t):
        if using_dataset:
            x = batches[step % MAX_LEN][0].to(device)
            x = emb_tokens(x)
        else:
            x = torch.randn(batch_size, seq_len, dim).to(device)

        with torch.no_grad():
            target = transformer_layer(x)[0]
        pred = mamba_layer(x)[0]
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # mamba_layer.apply_weight_regularization()
        writer.add_scalar("Loss_of_approximation", loss.item(), step)
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.6f}")
    
    save_mamba_layer(mamba_layer=mamba_layer, idx=number_of_layer)

def run(layer: int, max_t: int):
    path_to_config = f'./transformer_layer/model/layer{layer}'
    mamba_layer = get_mamba_layer(path_to_config)
    transformer_layer = get_transformer_layer(path_to_config)
    print(f'[INFO] Start of training process')
    training(mamba_layer=mamba_layer, transformer_layer=transformer_layer, number_of_layer=layer, layer_path=path_to_config, max_t=max_t)
    print(f'[INFO] End of training')

if __name__ == "__main__":
    fix_seeds()
    run(1, 7000)