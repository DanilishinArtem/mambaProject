import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Убедимся, что размерность делится на число голов
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads
        self.chunk_size = None

        # Основная Mamba
        self.mamba_core = Mamba(
            d_model=self.head_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # Система весов и смещений
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.shift_params = nn.Parameter(torch.zeros(num_heads))
        
        # Нормализация
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Проекционные слои
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Кеширование смещений
        self.shift_cache = {}

    def _build_shift_mask(self, L, device):
        H = self.num_heads
        chunk_size = L // H
        
        # Рассчитываем смещения для каждой головы
        shifts = (torch.sigmoid(self.shift_params) * chunk_size).int()
        mask = torch.zeros(H, L, device=device)
        
        for i in range(H):
            start = i * chunk_size + shifts[i]
            mask[i, start:start+chunk_size] = 1
            
        return mask

    def forward(self, hidden_states, residual=None):
        B, L, D = hidden_states.shape
        H = self.num_heads
        C = L // H
        
        # LayerNorm
        hidden_states, residual = layer_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            prenorm=True,
            residual=residual,
            is_rms_norm=True
        )
        
        # Проекция входа
        x = self.in_proj(hidden_states)
        
        # Правильное разделение на головы
        x = x.view(B, L, H, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, head_dim]
        
        # Создаем/получаем маску смещений
        cache_key = f"{L}_{H}"
        if cache_key not in self.shift_cache:
            self.shift_cache[cache_key] = self._build_shift_mask(L, x.device)
        
        shift_mask = self.shift_cache[cache_key]
        
        # Применяем смещения
        shifted = x * shift_mask.view(1, H, L, 1)
        
        # Обработка чанков
        outputs = []
        for i in range(H):
            # Извлекаем чанк для i-й головы
            chunk = shifted[:, i, :, :]  # [B, L, head_dim]
            
            # Вычисляем смещение
            start = i * C + int(self.shift_params[i].sigmoid() * C)
            active_chunk = chunk[:, start:start+C, :]
            
            # Обработка Mamba
            processed = self.mamba_core(active_chunk)
            
            # Восстанавливаем полный размер
            full_output = torch.zeros(B, L, self.head_dim, device=x.device, dtype=x.dtype)
            full_output[:, start:start+C, :] = processed
            outputs.append(full_output.unsqueeze(1))  # [B, 1, L, head_dim]
        
        # Объединяем результаты голов
        combined = torch.cat(outputs, dim=1)  # [B, H, L, head_dim]
        
        # Собираем выход
        combined = combined.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_proj(combined)
        
        return self.dropout(output), residual


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