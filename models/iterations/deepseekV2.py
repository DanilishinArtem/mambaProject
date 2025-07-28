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
    def __init__(self, dim, num_heads, dropout=0.0, linear_attn_r=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim
        self.linear_attn_r = linear_attn_r  # Проекционная размерность для линейного внимания

        # Нормализация
        self.norm = RMSNorm(dim)  # Используем оригинальную RMSNorm

        # Проекции QKV (объединенные для эффективности)
        self.qkv_proj = nn.Linear(dim, dim * 3 * num_heads)
        
        # Линейное внимание: проекции для ключей и значений
        self.linear_attn_q_projs = nn.ModuleList([nn.Linear(dim, linear_attn_r) for _ in range(num_heads)])
        self.linear_attn_k_projs = nn.ModuleList([nn.Linear(dim, linear_attn_r) for _ in range(num_heads)])

        # ОРИГИНАЛЬНЫЕ Mamba-блоки для каждой головы
        self.mamba_heads = nn.ModuleList([
            create_block(
                d_model=dim,
                d_intermediate=0,
                ssm_cfg={"d_state": 32, "layer": "Mamba2"},
                layer_idx=i
            ) for i in range(num_heads)
        ])

        # Гейт-механизм
        self.gate_proj = nn.Linear(dim, dim)
        self.gate_act = nn.Sigmoid()

        # Выходная проекция
        self.out_proj = nn.Linear(dim * num_heads, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x, residual=None):
        B, L, D = x.shape
        if residual is None:
            residual = x

        # Pre-norm
        x_norm = self.norm(x)

        # Объединенная проекция QKV
        qkv = self.qkv_proj(x_norm).view(B, L, 3, self.num_heads, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, dim]

        head_outs = []
        for i in range(self.num_heads):
            # Извлекаем данные для i-й головы
            qi = q[:, i]  # [B, L, dim]
            ki = k[:, i]
            vi = v[:, i]

            # Линейное внимание (O(L))
            q_proj = F.elu(self.linear_attn_q_projs[i](qi)) + 1  # [B, L, r]
            k_proj = F.elu(self.linear_attn_k_projs[i](ki)) + 1  # [B, L, r]
            
            # Вычисляем взвешенные значения через ассоциативность
            kv = torch.einsum('blk,blv->bkv', k_proj, vi)  # [B, r, dim]
            attn_out = torch.einsum('blk,bkv->blv', q_proj, kv)  # [B, L, dim]
            
            # Нормализация
            z = 1.0 / (torch.einsum('blk->bl', q_proj) + 1e-6)  # [B, L]
            attn_out = attn_out * z.unsqueeze(-1)  # [B, L, dim]

            # ОРИГИНАЛЬНЫЙ Mamba-блок
            mamba_out, _ = self.mamba_heads[i](attn_out)  # [B, L, dim]
            
            head_outs.append(mamba_out)

        # Гейт-механизм
        gate = self.gate_act(self.gate_proj(x_norm))  # [B, L, dim]
        gated_heads = [ho * gate for ho in head_outs]
        combined = torch.cat(gated_heads, dim=-1)  # [B, L, dim * num_heads]

        # Выход
        out = self.out_proj(combined)
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