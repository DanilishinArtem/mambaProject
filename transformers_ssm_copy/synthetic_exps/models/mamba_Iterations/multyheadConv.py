# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def default(val, default_val):
    return val if val is not None else default_val

class LinformerSelfAttention(nn.Module):
    def __init__(
        self, dim, seq_len, k=256, heads=8, dim_head=None,
        one_kv_head=False, share_kv=False, dropout=0.0
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by heads'

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = dim_head or (dim // heads)
        self.share_kv = share_kv

        # Проекции
        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)

        kv_dim = self.dim_head if one_kv_head else (self.dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(torch.empty(seq_len, k).uniform_(-0.02, 0.02))

        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(torch.empty(seq_len, k).uniform_(-0.02, 0.02))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.dim_head * heads, dim)

        # --- Каузальная маска для проецированной длины ---
        # seq_len × k → после проекции токен t видит только прошлое и себя
        proj_mask = torch.ones(seq_len, k, dtype=torch.bool)
        for i in range(seq_len):
            allowed_k = min(k, i + 1)  # сколько позиций в проекции можно видеть
            proj_mask[i, allowed_k:] = False
        self.register_buffer("causal_proj_mask", proj_mask, persistent=False)

    def forward(self, x, context=None):
        b, n, _ = x.shape
        h, d_h, k = self.heads, self.dim_head, self.k

        kv_input = x if context is None else context
        kv_len = kv_input.shape[1]

        # --- Q, K, V ---
        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)  # b,h,n,d_h
        k_lin = self.to_k(kv_input)
        v_lin = k_lin if self.share_kv else self.to_v(kv_input)

        proj_k = self.proj_k[:kv_len]  # подрезаем при коротких последовательностях
        k_proj = torch.einsum('bnd,nk->bkd', k_lin, proj_k)  # b,k,d_kv
        v_proj = torch.einsum('bnd,nk->bkd', v_lin, proj_k if self.share_kv else self.proj_v[:kv_len])

        # --- Разбиваем на головы ---
        k_proj = k_proj.reshape(b, k, h, d_h).transpose(1, 2)  # b,h,k,d_h
        v_proj = v_proj.reshape(b, k, h, d_h).transpose(1, 2)  # b,h,k,d_h

        # --- Attention ---
        dots = torch.einsum('bhnd,bhkd->bhnk', q, k_proj) * (d_h ** -0.5)

        # Каузальная маска в проекции
        causal_mask = self.causal_proj_mask[:n, :k]
        dots = dots.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnk,bhkd->bhnd', attn, v_proj)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


import torch.nn.functional as F
class CausalLinearCombinationLocal(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.dim = dim
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, bias=False)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.k - 1, 0))
        y = self.conv(x)
        return y.transpose(1, 2)    

import os
class MixerModel(nn.Module):
    def __init__(self, d_model: int, n_layer: int, d_intermediate: int, vocab_size: int,
                 heads: int = 8,
                 ssm_cfg=None, attn_layer_idx=None, attn_cfg=None,
                 norm_epsilon: float = 1e-5, rms_norm: bool = False,
                 initializer_cfg=None, fused_add_norm=False,
                 residual_in_fp32=False, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.n_layer = n_layer
        self.heads = heads

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # Слои: [n_layer][n_heads]
        self.layers = nn.ModuleList([
            nn.ModuleList([
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    **factory_kwargs,
                ) for _ in range(self.heads)
            ]) for _ in range(self.n_layer)
        ])

        # Кауза́льные фильтры [n_layer][n_heads]
        self.layer_filter = nn.ModuleList([
            nn.ModuleList([
                CausalLinearCombinationLocal(dim=d_model, k=16) for _ in range(self.heads)
            ]) for _ in range(self.n_layer)
        ])

        # Веса для агрегации голов внутри каждого слоя
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.heads)) for _ in range(self.n_layer)
        ])

        # Финальная нормализация
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # Инициализация весов
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

        self.counter = 0

    def batch_covariance(self, x: torch.Tensor, by_time: bool = True) -> torch.Tensor:
        """
        Вычисляет ковариацию.
        Если by_time=True → ковариация между временными шагами.
        Если False → ковариация между признаками.
        """
        if by_time:
            # Считаем ковариацию по оси seq_len
            # x: [batch, seq_len, dim] → [batch, dim, seq_len]
            x = x.transpose(1, 2)
        # Центрируем
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.matmul(x_centered, x_centered.transpose(-1, -2))
        cov = cov / (x.shape[-1] - 1)

        path = "/home/adanilishin/mambaProject/tensors/mamba_copy.pt"
        if not os.path.exists(path):
            torch.save(cov, path)
            print(f"[INFO] Covariance tensor saved at: {path}")
        return cov

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: [head.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                for head in layer]
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)  # [batch, seq_len, dim]
        residual = None
        self.counter += 1

        for idx_layer in range(self.n_layer):
            w = torch.softmax(self.weights[idx_layer], dim=0)  # [n_heads]
            total_heads_hidden_states = torch.zeros_like(hidden_states)

            for idx_head in range(self.heads):
                head_input = self.layer_filter[idx_layer][idx_head](hidden_states)
                head_output, _ = self.layers[idx_layer][idx_head](
                    head_input, None, inference_params=inference_params, **mixer_kwargs
                )
                total_heads_hidden_states += head_output * w[idx_head]

            # Residual connection (один раз после агрегации всех голов)
            residual = (total_heads_hidden_states + hidden_states) if residual is None else (total_heads_hidden_states + residual)
            hidden_states = residual

        # Считаем ковариацию в определённые моменты
        if self.counter in (2000, 5000):
            self.batch_covariance(hidden_states, by_time=True)

        # Финальная нормализация
        if not self.fused_add_norm:
            hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=None,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        return hidden_states

class MambaPlusPlusML(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)