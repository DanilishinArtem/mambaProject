from evalscope import run_task, TaskConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mamba2 import get_mamba_by_config
from mamba_ssm.models.config_mamba import MambaConfig


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = 128
    n_layer = 4
    state_dim = 16

def run():
    DATASETS_ARGS = {
        "arc": {
            "local_path": "/home/adanilishin/mambaProject/evalscope/datasets/arc_challenge/validation",
            "few_shot_num": 0,
            "few_shot_random": False,
        },
    }

    # укажи свою модель
    model_path = "/home/adanilishin/mambaProject/evalscope/models/mamba"  # путь к твоему checkpoint
    model_name = "Mamba2-1.3B"

    # конфигурация генерации
    generation_config = {
        "do_sample": True,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 50,
        "n": 1,
    }

    test_datasets = ["arc"]
    config = Config
    config = MambaConfig(
        d_model=config.d_model,
        n_layer=config.n_layer,
        rms_norm=True,
        ssm_cfg={"d_state": config.state_dim, "layer" : "Mamba2", "use_mem_eff_path": False},
        vocab_size=len(config.tokenizer)
    )
    model = get_mamba_by_config(config, "/home/adanilishin/mambaProject/evalscope/models/torch/mamba.pt")
    model.api.model.eval() 

    task_config = TaskConfig(
        model=model,
        datasets=test_datasets,
        dataset_hub="Local",
        model_args={
            "revision": "master",
            "precision": "torch.float",
            "device_map": "cuda",
        },
        generation_config=generation_config,
        dataset_args={name: DATASETS_ARGS[name] for name in test_datasets},
        work_dir="./eval_results",
        eval_batch_size=1,
        repeats=1,
    )
    # запускаем
    eval_results = run_task(task_cfg=task_config)
    print(eval_results)


if __name__ == "__main__":
    run()