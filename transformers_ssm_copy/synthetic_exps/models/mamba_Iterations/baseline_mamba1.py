import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.mambas = Mamba(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        self.norm = RMSNorm(dim)

    def forward(self, hidden_states, residual=None, padding_mask=None):
        hidden_states, residual = layer_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            prenorm=True,
            residual=residual,
            is_rms_norm=True
        )
        hidden_states = self.mambas(hidden_states)
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