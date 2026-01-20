import os
from models import (
        LSTM,
        GPTNeoXAlibiForCausalLM,
        GPTNeoXHardAlibiForCausalLM,
        GPTNeoXNoPEForCausalLM,
        )
from transformers import  GPTNeoXForCausalLM, GPTNeoXConfig


def get_model(args, tokenizer):
    if args.model in ["T_nope","T_rope","T_alibi"]:
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=args.heads,
                    num_hidden_layers=args.layers,
                    vocab_size=len(tokenizer),
                    )
    elif args.model == "T_hard_alibi":
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=args.heads,
                    num_hidden_layers=args.layers,
                    num_masked_heads=args.num_masked_heads,
                    vocab_size=len(tokenizer),
                    )
    
    if args.model=="T_rope":
        model = GPTNeoXForCausalLM(config)
    elif args.model=="T_nope":
        model = GPTNeoXNoPEForCausalLM(config)
    elif args.model=="T_alibi":
        model = GPTNeoXAlibiForCausalLM(config)
    elif args.model=="T_hard_alibi":
        model = GPTNeoXHardAlibiForCausalLM(config)
    elif args.model=="mamba":
        from lib.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        d_model = 768
        config = MambaConfig(
            d_model=d_model,
            d_intermediate=3072,
            n_layer=12,
            rms_norm=True,
            ssm_cfg={"d_state": 32, "layer": "Mamba2"},
            # ssm_cfg={"d_state": 32, "expand": 1, "headdim": d_model // 2, "layer": "Mamba2"},
            vocab_size=len(tokenizer)
        )
        model = MambaLMHeadModel(config)
    return model