import torch
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks as task_registry
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import json
import os
from lib.mamba import create_block

def load_mamba_layer(idx: int):
    layer = None
    layer_path = f'./mamba_layer/model/layer{idx}'
    if os.path.exists(layer_path):
        with open(f"{layer_path}/config.json", "r") as f:
            cfg = json.load(f)
        layer = create_block(
            d_model=cfg["hidden_size"],
            d_intermediate=0,
            ssm_cfg={"d_state": cfg["d_state"], "layer": cfg["layer_type"]},
            layer_idx=idx
        )
        state = torch.load(f"{layer_path}/pytorch_model.bin", map_location="cpu")
        layer.load_state_dict(state)
    return layer

def get_model():
    name = 'facebook/opt-125m'
    config = AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_config(config)
    # model = AutoModelForCausalLM.from_pretrained(name)
    model_name = './transformer_layer/model/'
    state_dict = torch.load(f'{model_name}/full_model/pytorch_model.bin')
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(f'{model_name}/tokenizer')
    full_config = model.config
    for idx in range(len(model.model.decoder.layers)):
        new_layer = None
        new_layer = load_mamba_layer(idx + 1)
        if new_layer == None:
            path_to_layer = f'./transformer_layer/model/layer{idx+1}'
            new_layer = OPTDecoderLayer(full_config, layer_idx=idx)
            state = torch.load(
                f"{path_to_layer}/pytorch_model.bin",
                map_location="cuda",
                weights_only=True
            )
            new_layer.load_state_dict(state)
            model.model.decoder.layers[idx].load_state_dict(state)
        else:
            model.model.decoder.layers[idx] = new_layer
        print(f'[INFO] Replaced layer {idx}')
    print(model)
    # model = model.to(torch.float32)
    model = model.cuda()
    return model, tokenizer

def main():
    model, tokenizer = get_model()
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=8,
        trust_remote_code=True
    )
    # tasks = ["arc_easy", "fda", "swde"]
    tasks = ["fda"]
    for task in tasks:
        results = None
        tm = task_registry.TaskManager()
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task,
            task_manager=tm
        )
        print(make_table(results))        
    
    
if __name__ == "__main__":
    main()