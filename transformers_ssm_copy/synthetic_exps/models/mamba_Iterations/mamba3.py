import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaPlusPlusLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Mamba heads с разными параметрами
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16,  # Уменьшено для скорости
                d_conv=4,
                expand=2,
            )
            for _ in range(num_heads)
        ])
        
        # Нормализации
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_heads)
        ])
        
        # Веса голов с инициализацией
        self.head_weights = nn.Parameter(torch.randn(num_heads))
        
        # Эффективные линейные преобразования
        self.W_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Оптимизированный FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, emb):
        B, L, D = emb.shape
        H = self.num_heads
        
        head_outputs = []
        for h in range(H):
            stride = h + 1
            # Эффективная субдискретизация
            indices = torch.arange(0, L, stride, device=emb.device)
            x_sub = emb[:, indices, :]
            
            # Обработка Mamba
            x_norm = self.norms[h](x_sub)
            y = self.mambas[h](x_norm)
            
            # Быстрое восстановление через индексацию
            restored = torch.zeros(B, L, D, device=emb.device, dtype=emb.dtype)
            restored[:, indices, :] = y
            
            head_outputs.append(restored)
        
        # Векторизованная агрегация
        weights = F.softmax(self.head_weights, dim=0)
        aggregated = torch.sum(
            torch.stack(head_outputs) * weights.view(H, 1, 1, 1),
            dim=0
        )
        
        # Residual connection
        res = self.W_out(aggregated)
        out = emb + self.dropout(res)
        
        # FFN блок
        ffn_out = self.ffn(out)
        return self.norm_out(out + ffn_out)


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.layers = nn.ModuleList([
            MambaPlusPlusLayer(dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Весовая инициализация
        nn.init.normal_(self.pos_embed, std=0.01)
        self.lm_head.weight = self.embed.weight  # Weight tying
        
    def forward(self, input_ids):
        emb = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1), :]
        x = self.dropout(emb)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return {"logits": self.lm_head(x)}