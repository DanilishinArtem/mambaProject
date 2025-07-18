import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
import numpy as np

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16 + i,
                d_conv=4,
                expand=2,
            )
            for i in range(num_heads)
        ])
        # self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.norm = RMSNorm(dim)
        self.perceptron = nn.Linear(num_heads * dim, dim)

    def forward(self, hidden_states, residual=None, padding_mask=None):
        B, L, D = hidden_states.shape
        H = self.num_heads

        hidden_states, residual = layer_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            prenorm=True,
            residual=residual,
            is_rms_norm=True
        )

        head_outputs = []
        for h in range(H):
            stride = h + 1  # 1 для первой головы, 2 — для второй, и т.д.
            x_subsampled = hidden_states[:, ::stride, :]  # (B, L_h, D)
            y = self.mambas[h](x_subsampled)  # (B, L_h, D)
            # Вставим обратно в тензор размера (B, L, D)
            expanded = torch.zeros(B, L, D, device=hidden_states.device, dtype=hidden_states.dtype)
            expanded[:, ::stride, :] = y
            head_outputs.append(expanded)        

        stacked = torch.stack(head_outputs, dim=0).permute(1, 2, 0, 3).reshape(B, L, H * D)
        hidden_states = self.perceptron(stacked)
        return hidden_states, residual

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
        return {"logits": logits}