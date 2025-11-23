import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
BASE_DIR = "./model"

def run(name: str):
    config = AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    os.makedirs(f'{BASE_DIR}/full_model', exist_ok=True)
    os.makedirs(f'{BASE_DIR}/tokenizer', exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, f'{BASE_DIR}/full_model/pytorch_model.bin')
    tokenizer.save_pretrained(f'{BASE_DIR}/tokenizer')
    layers = model.model.decoder.layers
    print("Количество слоёв:", len(layers))
    for i, layer in enumerate(layers):
        layer_dir = os.path.join(BASE_DIR, f"layer{i+1}")
        os.makedirs(layer_dir, exist_ok=True)
        torch.save(layer.state_dict(), os.path.join(layer_dir, "pytorch_model.bin"))
        with open(os.path.join(layer_dir, "config.json"), "w") as f:
            f.write(config.to_json_string())
        print(f"Сохранён слой {i+1} → {layer_dir}")

if __name__ == "__main__":
    model_name = 'facebook/opt-125m'
    run(model_name)