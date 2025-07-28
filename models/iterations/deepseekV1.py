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
    def __init__(self, dim, num_heads, dropout=0.0, linear_attn_r=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.linear_attn_r = linear_attn_r  # Проекция в r

        # LayerNorm для стабильности
        self.norm = nn.LayerNorm(dim)

        # Объединённая QKV (не используем для линear-attn, но можно оставить)
        self.qkv_proj = nn.Linear(dim, dim * 3)

        # Проекции для линear-attn: q, k, v → размер r
        self.linear_attn_q_projs = nn.ModuleList([nn.Linear(dim, linear_attn_r) for _ in range(num_heads)])
        self.linear_attn_k_projs = nn.ModuleList([nn.Linear(dim, linear_attn_r) for _ in range(num_heads)])
        self.linear_attn_v_projs = nn.ModuleList([nn.Linear(dim, linear_attn_r) for _ in range(num_heads)])

        # Mamba-блокы (заменили на depthwise+pointwise conv, как в твоём примере)
        self.mamba_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(linear_attn_r, linear_attn_r, kernel_size=3, padding=1, groups=linear_attn_r),
                nn.GELU(),
                nn.Conv1d(linear_attn_r, dim, kernel_size=1),
            )
            for _ in range(num_heads)
        ])

        # Гейтирование
        self.gate_proj = nn.Linear(dim, dim)
        self.gate_act = nn.Sigmoid()

        # Выходная проекция собирает num_heads*dim → dim
        self.out_proj = nn.Linear(dim * num_heads, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x, residual=None):
        B, L, D = x.shape
        if residual is None:
            residual = x

        # Pre-norm
        x_norm = self.norm(x)

        head_outs = []
        for i in range(self.num_heads):
            qi = x_norm
            ki = x_norm
            vi = x_norm

            # Проекции q,k,v → [B, L, r]
            q_proj = F.gelu(self.linear_attn_q_projs[i](qi))  # (B, L, r)
            k_proj = F.gelu(self.linear_attn_k_projs[i](ki))  # (B, L, r)
            v_proj = F.gelu(self.linear_attn_v_projs[i](vi))  # (B, L, r)

            # Ассоциативная linear-attn
            # kv: [B, r, r]
            kv = torch.bmm(k_proj.transpose(1, 2), v_proj)
            # attn_out: [B, L, r]
            attn_out = torch.bmm(q_proj, kv)

            # Нормировка по сумме q
            z = 1.0 / (q_proj.sum(dim=-1, keepdim=True) + 1e-6)  # (B, L, 1)
            attn_out = attn_out * z

            # Mamba-блок через conv: нужно [B, r, L]
            mamba_in = attn_out.transpose(1, 2)        # (B, r, L)
            mamba_out = self.mamba_heads[i](mamba_in)  # (B, dim, L)
            mamba_out = mamba_out.transpose(1, 2)      # (B, L, dim)

            head_outs.append(mamba_out)

        # Гейтирование
        gate = self.gate_act(self.gate_proj(x_norm))  # (B, L, dim)
        gated_heads = [h * gate for h in head_outs]   # по-головное умножение

        # Конкатенация и выход
        combined = torch.cat(gated_heads, dim=-1)     # (B, L, num_heads*dim)
        out = self.out_proj(combined)                 # (B, L, dim)
        out = residual + self.dropout(out) * self.layer_scale
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