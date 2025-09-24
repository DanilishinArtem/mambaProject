import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = 128
    n_layer = 4
    state_dim = 16

def get_model(config):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    config = MambaConfig(
        d_model=config.d_model,
        n_layer=config.n_layer,
        ssm_cfg={"d_state": config.state_dim, "layer" : "Mamba2", "use_mem_eff_path": False},
        vocab_size=len(config.tokenizer)
    )
    model = MambaLMHeadModel(config)
    return model


def save_model():
    path_to_torch = "/home/adanilishin/mambaProject/evalscope/models/torch"
    path_to_safetensors = "/home/adanilishin/mambaProject/evalscope/models/mamba"
    model = get_model(Config)
    print(f'[DEBUG] Model created')
    torch.save(model.state_dict(), f'{path_to_torch}/mamba.pt')
    print(f'[DEBUG] Model saved')
    state_dict = torch.load(f'{path_to_torch}/mamba.pt')
    model.load_state_dict(state_dict)
    print(f'[DEBUG] Model loaded from torch')
    model.save_pretrained(f'{path_to_safetensors}/')
    print(f'[DEBUG] Model saved to mamba in format safetensors')
    Config.tokenizer.save_pretrained(f'{path_to_safetensors}/')
    print(f'[DEBUG] Tokenizer saved to mamba in format safetensors')
    print(f'[DEBUG] Start try load model by using AutoModelForCausalLM')


if __name__ == "__main__":
    save_model()