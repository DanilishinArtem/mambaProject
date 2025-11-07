# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, attention_mask: Optional[Tensor]=None, position_ids: Optional[torch.LongTensor]=None, cache_position: Optional[torch.Tensor]=None, inference_params=None, **mixer_kwargs
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if not isinstance(self.mixer, (Mamba, Mamba2)):
            hidden_states, self.mixer(
                    hidden_states=hidden_states,
                    past_key_values=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    cache_position=cache_position,
                    **mixer_kwargs,
                )
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
