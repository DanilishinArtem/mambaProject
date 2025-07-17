import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_heads)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_heads)
        ])

        # 🔸 Обучаемые веса агрегации для каждой головы
        self.head_weights = nn.Parameter(torch.ones(num_heads))

        self.C = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, emb, padding_mask=None):
        B, L, D = emb.shape
        H = self.num_heads

        head_outputs = []

        for h in range(H):
            stride = h + 1  # 1 для первой головы, 2 — для второй, и т.д.
            x_subsampled = emb[:, ::stride, :]  # (B, L_h, D)
            x_norm = self.norms[h](x_subsampled)
            y = self.mambas[h](x_norm)  # (B, L_h, D)

            # Вставим обратно в тензор размера (B, L, D)
            expanded = torch.zeros(B, L, D, device=emb.device, dtype=emb.dtype)
            expanded[:, ::stride, :] = y
            head_outputs.append(expanded)

        # 🔸 Взвешенное суммирование с обучаемыми весами
        weights = torch.softmax(self.head_weights, dim=0)  # (H,)
        stacked = torch.stack(head_outputs, dim=0)         # (H, B, L, D)
        weighted = (weights.view(H, 1, 1, 1) * stacked).sum(dim=0)  # (B, L, D)

        # Остальная часть блока
        c_out = self.C(weighted)
        w_out = self.W_out(emb)
        res = c_out + w_out

        z = self.norm1(emb + self.dropout(res))
        ffn_out = self.ffn(z)
        return self.norm2(z + ffn_out)


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaPlusPlus_layer(dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None):
        emb = self.embed(input_ids)
        x = self.dropout(emb)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits}