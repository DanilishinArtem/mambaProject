import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.mambas = nn.ModuleList([
            Mamba(
                d_model=self.head_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                # dt_min=0.001 * (1.5 ** h),
                # dt_max=0.01 * (1.5 ** h),
                # dt_scale=1.0,
            )
            for h in range(num_heads)
        ])

        # Hyena-style gating per head
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.SiLU()
            )
            for _ in range(num_heads)
        ])

        # Depth-wise conv kernel per head
        self.kernels = nn.ModuleList([
            nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=self.head_dim)
            for _ in range(num_heads)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, hidden_states, residual=None, padding_mask=None):
        B, L, D = hidden_states.shape
        H = self.num_heads
        Dh = self.head_dim
        # Normalization
        hidden_states, residual = layer_norm_fn(
            hidden_states, self.norm.weight, self.norm.bias,
            prenorm=True, residual=residual, is_rms_norm=True
        )

        # Split into heads: (B, L, D) → (B, H, L, Dh)
        hs = hidden_states.view(B, L, H, Dh).transpose(1, 2)

        head_outputs = []
        for h in range(H):
            x_h = hs[:, h, :, :]  # (B, L, Dh)

            # Core processing
            y = self.mambas[h](x_h)  # (B, L, Dh)

            # Depth-wise conv
            y = self.kernels[h](y.transpose(1, 2)).transpose(1, 2)  # (B, L, Dh)
            gate = self.gates[h](x_h)
            y = y * gate
            head_outputs.append(y)

        # Concatenate head outputs: (B, H, L, Dh) → (B, L, D)
        mixed = torch.cat(head_outputs, dim=-1)
        hidden_states = mixed.view(B, L, Dh * H)

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
        return logits