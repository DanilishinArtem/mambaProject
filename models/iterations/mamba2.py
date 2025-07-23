import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads=None, kernel_size=3, dropout=0.0):
        super().__init__()
        expand = 4
        dt_hidden = 64

        self.dim = dim
        self.expand = expand
        self.d_inner = expand * dim  # ===> Теперь d_inner

        # 1) Предобработка: Pre‑Norm
        self.norm = RMSNorm(dim)

        # 2) dt_net: из (dim) → dt_hidden → (d_inner)
        self.dt_net = nn.Sequential(
            nn.Linear(dim, dt_hidden),
            nn.SiLU(),
            nn.Linear(dt_hidden, self.d_inner),  # <-- важно: self.d_inner
        )

        # 3) SSM‑блок
        self.ssm = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=kernel_size,
            expand=expand,
            dt_init="constant",
            dt_scale=0.0,
        )
        # инициализируем bias правильно
        nn.init.zeros_(self.ssm.dt_proj.bias)  # shape == (d_inner,)

        # 4) Gating и Mixing
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """
        x: (B, L, dim)
        residual: same shape or None
        """
        if residual is None:
            residual = x

        # Pre‑norm
        x_norm, _ = layer_norm_fn(
            x, self.norm.weight, self.norm.bias,
            prenorm=True, residual=None, is_rms_norm=True
        )

        B, L, D = x_norm.shape

        # 1) Собираем глобальный summary по токенам (или можно батч+позиция)
        #    у нас summary.shape == (B, dim)
        summary = x_norm.mean(dim=1)  # (B, dim)

        # 2) dt_net: (B, dim) -> (B, d_inner)
        dt_bias_batch = self.dt_net(summary)  # (B, d_inner)

        # 3) Агрегируем по батчу: вектор (d_inner,)
        dt_bias = dt_bias_batch.mean(dim=0)  # (d_inner,)

        # 4) Копируем в bias SSM
        with torch.no_grad():
            self.ssm.dt_proj.bias.copy_(dt_bias)

        # 5) Пропускаем через SSM
        ssm_out = self.ssm(x_norm)  # (B, L, dim)

        # 6) Gating
        gate = torch.sigmoid(self.gate(x_norm))  # (B, L, dim)
        y = ssm_out * gate

        # 7) Channel mixing + dropout + residual
        y = self.out_proj(y)
        y = self.dropout(y)
        return y + residual, residual

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