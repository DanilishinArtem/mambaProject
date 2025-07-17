import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPlusPlusLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Упрощенные Mamba головы
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=32,  # Оптимально для производительности
                d_conv=4,
                expand=2,
            )
            for _ in range(num_heads)
        ])
        
        # Эффективные нормализации
        self.norm = nn.LayerNorm(dim)  # Общая нормализация
        
        # Динамические веса голов
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.W_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Оптимизированный FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, emb):
        B, L, D = emb.shape
        H = self.num_heads
        
        # Общая нормализация перед обработкой
        emb_norm = self.norm(emb)
        
        head_outputs = []
        for h in range(H):
            stride = h + 1
            # Эффективная субдискретизация через индексацию
            indices = torch.arange(0, L, stride, device=emb.device)
            x_sub = emb_norm[:, indices, :]
            
            # Обработка Mamba
            y = self.mambas[h](x_sub)
            
            # Быстрое восстановление через scatter
            restored = torch.zeros(B, L, D, device=emb.device, dtype=emb.dtype)
            restored[:, indices, :] = y
            head_outputs.append(restored)
        
        # Параллельная агрегация
        stacked = torch.stack(head_outputs, dim=0)
        weights = torch.softmax(self.head_weights, dim=0).view(H, 1, 1, 1)
        aggregated = (stacked * weights).sum(dim=0)
        
        # Residual connection
        res = self.W_out(aggregated)
        out = emb + self.dropout(res)
        
        # FFN блок
        ffn_out = self.ffn(out)
        return out + ffn_out


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            MambaPlusPlusLayer(dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Весовая инициализация
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.weight  # Weight tying
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids):
        # Только эмбеддинги токенов, без позиционных эмбеддингов
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return {"logits": self.lm_head(x)}