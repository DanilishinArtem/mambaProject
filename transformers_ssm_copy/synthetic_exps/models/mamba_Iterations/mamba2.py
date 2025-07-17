import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaPlusPlusLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim

        # Каждый "голова" — независимый Mamba-блок
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=self.head_dim,
                d_state=32,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_heads)
        ])

        self.norm_mixer = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        # после concat всех голов делаем projection обратно в dim
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.norm_mixer(x)

        # Каждая голова работает на нормализованном входе
        head_outputs = [mamba(x_norm) for mamba in self.mambas]  # List of (B, L, D)
        x_cat = torch.cat(head_outputs, dim=-1)  # (B, L, H*D)
        mixed = self.proj(x_cat)  # (B, L, D)

        x = x + self.dropout(mixed)

        # FFN
        x_norm2 = self.norm_ffn(x)
        ffn_out = self.ffn(x_norm2)
        return x + ffn_out


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaPlusPlusLayer(dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :L, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
