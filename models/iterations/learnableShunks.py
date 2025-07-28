import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block
import math

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.D = dim
        self.H = num_heads
        self.Dh = dim // num_heads  # per-head dimension

        # входная и выходная проекции
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.post_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Mamba-блоки по головам с разными seed для разнообразия
        self.mamba_heads = nn.ModuleList([
            create_block(
                d_model=self.Dh,
                d_intermediate=0,
                ssm_cfg={"d_state": 32, "layer": "Mamba2", "dt_init": "random", "dt_init_floor": 1e-5},
                layer_idx=i
            ) for i in range(self.H)
        ])

        # Гейты по головам
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.Dh, self.Dh),
                nn.SiLU(),
                nn.Linear(self.Dh, self.Dh),
                nn.Sigmoid()
            ) for _ in range(self.H)
        ])

    def forward(self, x, residual=None):
        B, L, _ = x.shape
        if residual is None:
            residual = x

        # Pre-norm + входная проекция
        x = self.post_norm(x)
        x = self.in_proj(x)  # (B, L, D)

        # Split to heads
        x_heads = x.view(B, L, self.H, self.Dh).transpose(1, 2)  # (B, H, L, Dh)

        head_outs = []
        for i, (head, gate) in enumerate(zip(self.mamba_heads, self.gates)):
            xi = x_heads[:, i]  # (B, L, Dh)
            # learnable or random shift
            shift = (L // self.H) * i  # fixed cyclic shift to diversify receptive fields
            xi = torch.roll(xi, shifts=shift, dims=1)

            yi, _ = head(xi)      # (B, L, Dh)
            gi = gate(xi)         # (B, L, Dh)
            head_outs.append(yi * gi)

        # concat heads
        heads = torch.stack(head_outs, dim=2)           # (B, L, H, Dh)
        concat = heads.view(B, L, self.D)               # (B, L, D)

        # вывод
        out = self.out_proj(concat) / math.sqrt(self.Dh)
        out = self.dropout(out)
        out = self.post_norm(out + residual)

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