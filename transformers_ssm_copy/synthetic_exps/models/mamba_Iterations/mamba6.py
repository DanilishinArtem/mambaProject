import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import math

class MambaPlusPlusLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, 
                 norm_epsilon=1e-5, fused_add_norm=False,
                 residual_in_fp32=False, layer_idx=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Инициализация нормализации
        self.norm = nn.LayerNorm(dim, eps=norm_epsilon)
        
        # Mamba heads с разными параметрами
        self.mamba_heads = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16 + i*4,  # Разные размеры состояния для разнообразия
                d_conv=4,
                expand=2,
                layer_idx=layer_idx,
            ) for i in range(num_heads)
        ])
        
        # Механизм внимания для агрегации голов
        self.head_aggregation = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,  # Используем 1 голову для агрегации
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Оптимизированный FFN с GLU
        self.ffn = GatedMLP(dim, hidden_features=dim * 2)
        self.ffn_dropout = nn.Dropout(dropout)
        
        # Нормализация после FFN
        self.norm_ffn = nn.LayerNorm(dim, eps=norm_epsilon)

    def forward(self, hidden_states, residual=None, inference_params=None):
        B, L, D = hidden_states.shape
        
        # Нормализация перед обработкой
        hidden_states_norm = self.norm(hidden_states)
        
        # Обработка Mamba-головами
        head_outputs = []
        for mamba_head in self.mamba_heads:
            # Каждая голова обрабатывает ВСЮ последовательность
            y = mamba_head(hidden_states_norm, inference_params=inference_params)
            head_outputs.append(y)
        
        # Конкатенация выходов голов
        all_heads = torch.stack(head_outputs, dim=1)  # [B, num_heads, L, D]
        
        # all_heads: [B, num_heads, L, D] → [B, L, num_heads * D]
        all_heads_flat = all_heads.permute(0, 2, 1, 3).reshape(B, L, -1)  # [B, L, num_heads * D]

        # query остается как [B, L, D]
        query = hidden_states_norm  # [B, L, D]

        # Применим проекцию, чтобы dim совпадал
        projection = nn.Linear(self.num_heads * D, D).to(all_heads.device)
        key_value = projection(all_heads_flat)

        aggregated, _ = self.head_aggregation(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False
        )
        aggregated = aggregated.squeeze(1)  # [B, L, D]
        
        
        # Residual connection
        if residual is None:
            residual = hidden_states
        else:
            residual = residual.to(aggregated.dtype)
        
        # Применяем residual
        if self.fused_add_norm:
            raise NotImplementedError("Fused add-norm не реализован")
        else:
            hidden_states = residual + self.dropout(aggregated)
        
        # FFN блок
        ffn_out = self.ffn(hidden_states)
        ffn_out = self.ffn_dropout(ffn_out)
        
        # Residual + FFN
        if self.fused_add_norm:
            raise NotImplementedError("Fused add-norm не реализован")
        else:
            out = hidden_states + ffn_out
            out = self.norm_ffn(out)
        
        return out, hidden_states.detach() if self.residual_in_fp32 else out


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, 
                 max_seq_len, dropout=0.1, norm_epsilon=1e-5,
                 fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Создаем слои с многоголовой Mamba
        self.layers = nn.ModuleList([
            MambaPlusPlusLayer(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                norm_epsilon=norm_epsilon,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        # Финальная нормализация
        self.norm_f = nn.LayerNorm(dim, eps=norm_epsilon)
        
        # Голова для языкового моделирования
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Связываем веса эмбеддингов и lm_head
        self.lm_head.weight = self.embed.weight
        
        # Применяем инициализацию весов
        self.apply(self._init_weights)
        self._apply_residual_init()

    def _init_weights(self, module):
        # Инициализация как в GPT-2
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _apply_residual_init(self):
        # Инициализация с учетом глубины сети
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "weight" in name and ("mamba" in name or "ffn" in name):
                    if param.ndim > 1:
                        nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * len(self.layers)))

    def forward(self, input_ids, inference_params=None):
        # Только эмбеддинги токенов
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=inference_params)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return {"logits": logits}