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
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        # # Regular mamba2
        # config = MambaConfig(
        #     d_model=args.hidden_size,
        #     n_layer=args.layers,
        #     d_intermediate=args.hidden_size*4,
        #     ssm_cfg={"d_state": 32, "layer": "Mamba2"},
        #     vocab_size=len(tokenizer)
        # )
        # Hybrid mamba2
        config = MambaConfig(
            d_model=768,
            d_intermediate=3072,
            n_layer=12,
            rms_norm=True,
            ssm_cfg={"d_state": 32, "layer": "Mamba2"},
            # attn_layer_idx=[11],
            # attn_layer_idx=[4,5,6,7],
            # attn_layer_idx=list(range(12)),
            # attn_cfg={},
            vocab_size=len(tokenizer)
        )
        model = MambaLMHeadModel(config)

    elif args.model=="lstm":
        model = LSTM(
                embedding_dim=args.hidden_size,
                vocab_size=len(tokenizer),
                num_layers=args.layers,
                dropout_rate=0.65
                )
    elif args.model=="mambapp":
        from models.mambapp import MambaPlusPlusML
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": 32, "layer" : "Mamba2", "headdim": 4, "expand": 2},
            vocab_size=len(tokenizer)
        )
        model = MambaPlusPlusML(config)
    elif args.model=="linformer":
        from linear_attention_transformer import LinearAttentionTransformerLM
        model = LinearAttentionTransformerLM(
            num_tokens = len(tokenizer),
            dim = args.hidden_size,
            heads = 64,
            depth = args.layers,
            max_seq_len = 8192
        )
    return model