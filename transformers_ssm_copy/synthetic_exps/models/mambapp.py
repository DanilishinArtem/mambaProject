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
class HybridPrefixFilter(nn.Module):
    """
    Гибридный каузальный фильтр (пер-канальный).

    - conv_small: короткое ядро, обучаемое — детектирует локальные пики/паттерны
    - conv_large: длинное ядро, инициализированное как экспоненциальный (Bayesian-like) фильтр
    - gate: учится балансировать вклад малого и большого фильтров

    Вход/выход: (B, L, D)
    """

    def __init__(
        self,
        dim,
        k_small=8,
        k_large=32,
        q=1e-3,
        r=0.5,
        decay=1.0,
        learnable_steady_init=True,
        eps=1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.k_small = k_small
        self.k_large = k_large
        self.eps = eps

        # depthwise convs (groups=dim) — каждый канал отдельно
        self.conv_small = nn.Conv1d(dim, dim, kernel_size=k_small, bias=False, groups=dim)
        self.conv_large = nn.Conv1d(dim, dim, kernel_size=k_large, bias=False, groups=dim)

        # per-dim gate logit (initial 0 => equal mixing)
        self.gate_logit = nn.Parameter(torch.zeros(dim))

        # init small conv as Xavier (flexible detector)
        nn.init.xavier_uniform_(self.conv_small.weight)

        # compute steady-state alpha/beta (to initialize large conv as exponential)
        q_t = torch.full((dim,), q)
        r_t = torch.full((dim,), r)
        disc = (q_t * q_t + 4 * q_t * r_t).clamp_min(eps)
        P_inf = (-q_t + torch.sqrt(disc)) / 2.0
        prior_prime = P_inf + q_t
        K = prior_prime / (prior_prime + r_t)
        alpha = ((1.0 - K) * float(decay)).clamp(min=0.0, max=0.999999)
        beta = K

        # build exponential kernels length k_large per-dim
        l_idx = torch.arange(k_large, dtype=torch.float32)
        exp_weights = (alpha.unsqueeze(1) ** ((k_large - 1) - l_idx.view(1, -1))) * beta.unsqueeze(1)  # (dim, k_large)

        if learnable_steady_init:
            # conv.weight shape (out_channels, in_channels/groups, kernel)
            # for depthwise: (dim, 1, k)
            with torch.no_grad():
                self.conv_large.weight.data.copy_(exp_weights.unsqueeze(1))
        else:
            nn.init.xavier_uniform_(self.conv_large.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> returns (B, L, D)"""
        B, L, D = x.shape
        assert D == self.dim
        x_ = x.transpose(1, 2)  # (B, D, L)

        # pad left for causality
        xs = F.pad(x_, (self.k_small - 1, 0))
        ys = self.conv_small(xs)[:, :, :L]  # ensure (B,D,L)

        xl = F.pad(x_, (self.k_large - 1, 0))
        yl = self.conv_large(xl)[:, :, :L]

        gate = torch.sigmoid(self.gate_logit.view(1, D, 1))
        y = gate * ys + (1.0 - gate) * yl

        return y.transpose(1, 2)


# --- Rewritten MixerModel fragment with HybridPrefixFilter integrated ---
class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.fused_add_norm = fused_add_norm

        self.layers = nn.ModuleList(
            [
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
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

        # Replace BayesianPrefixFilter with HybridPrefixFilter (per-layer)
        # conv filter on hidden states
        self.layer_filter_params_biasian = nn.ModuleList(
            [HybridPrefixFilter(d_model, k_small=8, k_large=32) for _ in range(n_layer)]
        )

        # per-layer filters for dt (post in_proj) — uses nheads from each block's mixer
        # create one per layer using the layer's mixer nheads
        self.layer_dt_filters = nn.ModuleList([
            HybridPrefixFilter(block.mixer.nheads, k_small=3, k_large=9)
            for block in self.layers
        ])

        self.g_parameter = nn.ParameterList([
            nn.Parameter(torch.tensor(0.05), requires_grad=True) for _ in range(n_layer)
        ])

        self.counter = 0

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def batch_covariance(self, x: torch.Tensor, by_time: bool = True) -> torch.Tensor:
        if by_time:
            x = x.transpose(1, 2)
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.matmul(x_centered, x_centered.transpose(-1, -2))
        cov = cov / (x.shape[-1] - 1)

        path = "/home/adanilishin/mambaProject/tensors/mamba_copy.pt"
        if not os.path.exists(path):
            torch.save(cov, path)
            print(f"[INFO] Covariance tensor saved at: {path}")
        return cov

    @staticmethod
    def _compute_token_mask_topk_from_dt(layer_mixer, hidden_states, keep_ratio=0.8, dt_filter: nn.Module | None = None):
        with torch.no_grad():
            zxbcdt = layer_mixer.in_proj(hidden_states)  # (B, L, d_in_proj)
        nheads = layer_mixer.nheads
        dt = zxbcdt[..., -nheads:]

        # optionally filter raw dt BEFORE softplus (can also filter after — experiment)
        if dt_filter is not None:
            # dt_filter expects (B,L,D)
            dt = dt_filter(dt)

        dt_sp = F.softplus(dt + layer_mixer.dt_bias.view(1, 1, -1).to(dt.device))
        B, L, H = dt_sp.shape
        assert H == nheads
        k = max(1, int(L * float(keep_ratio)))
        dt_perm = dt_sp.permute(0, 2, 1)  # (B, H, L)
        if k >= L:
            return torch.ones(B, L, H, dtype=torch.bool, device=dt_sp.device)
        values, indices = torch.topk(dt_perm, k=k, dim=-1)
        mask_bhl = torch.zeros(B, H, L, dtype=torch.bool, device=dt_sp.device)
        mask_bhl.scatter_(2, indices, True)
        return mask_bhl.permute(0, 2, 1)

    @staticmethod
    def _compute_token_mask_threshold_from_dt(layer_mixer, hidden_states, g=0.1, dt_filter: nn.Module | None = None):
        zxbcdt = layer_mixer.in_proj(hidden_states)  # (B, L, d_in_proj)
        nheads = layer_mixer.nheads
        dt = zxbcdt[..., -nheads:]

        if dt_filter is not None:
            dt = dt_filter(dt)

        dt_sp = torch.nn.functional.softplus(dt + layer_mixer.dt_bias.view(1, 1, -1).to(dt.device))
        mask = dt_sp >= g
        return mask

    def forward(
        self,
        input_ids,
        inference_params=None,
        use_token_filter: bool = True,
        keep_ratio: float = 0.8,
        use_conv_filter: bool = True,
        **mixer_kwargs,
    ):
        hidden_states = self.embedding(input_ids)  # (B, L, d_model)
        residual = None
        self.counter += 1

        for idx_layer, block in enumerate(self.layers):
            # Optional conv smoothing applied to hidden states
            if use_conv_filter:
                hidden_states = self.layer_filter_params_biasian[idx_layer](hidden_states)

            token_mask = None
            if use_token_filter:
                layer_mixer = block.mixer
                # pass per-layer dt_filter (or None if you want no dt filtering)
                token_mask = self._compute_token_mask_threshold_from_dt(
                    layer_mixer,
                    hidden_states,
                    g=1e-10,
                    dt_filter=(self.layer_dt_filters[idx_layer] if use_conv_filter else None),
                )
                mixer_call_kwargs = dict(inference_params=inference_params, token_mask=token_mask, **mixer_kwargs)
            else:
                mixer_call_kwargs = dict(inference_params=inference_params, **mixer_kwargs)

            hidden_states, residual = block(hidden_states, residual, **mixer_call_kwargs)

        if self.counter == 5000 or self.counter == 2000:
            self.batch_covariance(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
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