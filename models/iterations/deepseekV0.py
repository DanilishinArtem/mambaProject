import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block
from einops import rearrange, repeat
import math
import torch.nn.functional as F

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # per-head model dimension

        # normalization
        self.norm = RMSNorm(dim)

        # projections for each head
        self.q_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])

        # single-head attention per head
        self.attn = nn.MultiheadAttention(embed_dim=self.head_dim, num_heads=1, dropout=dropout, batch_first=True)

        # per-head Mamba blocks: d_model=head_dim, d_state=32
        self.mamba_heads = nn.ModuleList([
            create_block(
                d_model=self.head_dim,
                d_intermediate=0,
                ssm_cfg={"d_state": 32, "layer": "Mamba2"},
                layer_idx=i
            ) for i in range(num_heads)
        ])

        # gate projection
        self.gate_proj = nn.Linear(dim, dim)
        self.gate_act = nn.Sigmoid()

        # output projection from concat of heads
        self.out_proj = nn.Linear(self.head_dim * num_heads, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x, residual=None):
        # x: (B, L, dim)
        B, L, D = x.shape
        assert D == self.dim

        # pre-norm
        x_norm = self.norm(x)
        if residual is None:
            residual = x

        # compute gate per feature
        gate = self.gate_act(self.gate_proj(x_norm))  # (B, L, D)
        # reshape gate to per-head shape
        gate_heads = gate.view(B, L, self.num_heads, self.head_dim)

        head_outs = []
        for i in range(self.num_heads):
            # project to head_dim
            qi = self.q_projs[i](x_norm)
            ki = self.k_projs[i](x_norm)
            vi = self.v_projs[i](x_norm)

            # cyclic shift for diversity
            shift = (L // self.num_heads) * i
            qi = torch.roll(qi, shifts=shift, dims=1)
            ki = torch.roll(ki, shifts=shift, dims=1)
            vi = torch.roll(vi, shifts=shift, dims=1)

            # single-head attention
            attn_out, _ = self.attn(qi, ki, vi)  # (B, L, head_dim)

            # Mamba block per head
            mamba_out, _ = self.mamba_heads[i](attn_out)

            # apply gate for this head
            gated = mamba_out * gate_heads[:, :, i, :]
            head_outs.append(gated)

        # concatenate heads
        combined = torch.cat(head_outs, dim=-1)  # (B, L, head_dim * num_heads)

        # output
        out = self.out_proj(combined)  # (B, L, dim)
        out = self.dropout(out) * self.layer_scale
        out = residual + out
        return out, residual


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            MambaPlusPlus_layer(dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        hidden_states = layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            is_rms_norm=isinstance(self.norm_f, RMSNorm)
        )
        logits = self.lm_head(hidden_states)
        return logits