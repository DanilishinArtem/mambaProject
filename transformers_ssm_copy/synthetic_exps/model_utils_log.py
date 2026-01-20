import os

from transformers import  GPTNeoXForCausalLM, GPTNeoXConfig
from lib.mambalibi_rope import GPTNeoXForCausalLM
from lib.mambalibi import GPTNeoXAlibiForCausalLM
from transformers import AutoModel, AutoConfig
from lib.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from reMamba.config_remamba import ReMambaConfig
from reMamba.ReMamba import ReMambaLMHeadModel

def get_model(args, tokenizer):
    # mamba, opt, alibi, alibi_rope, remamba
    model_name = 'remamba'


    if model_name == 'remamba':
        config = ReMambaConfig(
            d_model=768,
            d_intermediate=3072,
            n_layer=12,
            rms_norm=False,
            ssm_cfg={"d_state": 32, "layer": "Mamba2"},
            vocab_size=50272
        )
        model = ReMambaLMHeadModel(config)
    elif model_name == 'opt':
        model_id = 'facebook/opt-125m'
        config = AutoConfig.from_pretrained(model_id)
        config.torch_dtype = "float32"
        model = AutoModel.from_config(config)
    elif model_name == 'mamba':
        config = MambaConfig(
            d_model=768,
            d_intermediate=3072,
            n_layer=12,
            rms_norm=False,
            ssm_cfg={"d_state": 32, "layer": "Mamba2"},
            attn_layer_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            attn_cfg={
                "_name_or_path": "facebook/opt-125m",
                "activation_dropout": 0.0,
                "activation_function": "relu",
                "architectures": [
                    "OPTForCausalLM"
                ],
                "attention_dropout": 0.0,
                "bos_token_id": 2,
                "do_layer_norm_before": True,
                "dropout": 0.1,
                "eos_token_id": 2,
                "ffn_dim": 3072,
                "hidden_size": 768,
                "init_std": 0.02,
                "layerdrop": 0.0,
                "max_position_embeddings": 2048,
                "model_type": "opt",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 1,
                "prefix": "</s>",
                "torch_dtype": "float32",
                "transformers_version": "4.21.0.dev0",
                "use_cache": True,
                "vocab_size": 50272,
                "word_embed_proj_dim": 768
                },
            vocab_size=50272
        )
        model = MambaLMHeadModel(config)
    elif model_name == 'alibi':
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=4,
                    num_hidden_layers=args.layers,
                    vocab_size=len(tokenizer),
                    )
        model = GPTNeoXAlibiForCausalLM(config, kernel='mamba', modification=True)
    else:
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=4,
                    num_hidden_layers=args.layers,
                    vocab_size=len(tokenizer),
                    )
        model = GPTNeoXForCausalLM(config=config, kernel='mamba', modification=True)
    return model