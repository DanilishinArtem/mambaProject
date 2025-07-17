import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import math

class MambaPlusPlusLayer(nn.Module):
    def __init__(self, dim, num_heads, residual_in_fp32=False, layer_idx=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.residual_in_fp32 = residual_in_fp32

        # Mamba heads
        self.mamba_heads = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16 + i * 4,
                d_conv=4,
                expand=2,
                layer_idx=layer_idx,
            ) for i in range(num_heads)
        ])

        # FFN (оставим как есть, но без dropout)
        self.ffn = GatedMLP(dim, hidden_features=dim * 2)

    def forward(self, hidden_states, residual=None, inference_params=None):
        head_outputs = [
            mamba_head(hidden_states, inference_params=inference_params)
            for mamba_head in self.mamba_heads
        ]

        # Простая агрегация — среднее по головам
        aggregated = torch.stack(head_outputs, dim=0).mean(dim=0)  # [B, L, D]

        if residual is None:
            residual = hidden_states
        else:
            residual = residual.to(aggregated.dtype)

        hidden_states = residual + aggregated

        ffn_out = self.ffn(hidden_states)
        out = hidden_states + ffn_out

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
        x = self.embed(input_ids)
        residual_from_input = x  # Сохраним skip изначальный
        x = self.dropout(x)  # отключи dropout в init

        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=inference_params)

        # Прямая skip-коннекция из embedding
        x = x + residual_from_input

        logits = self.lm_head(x)
        return {"logits": logits}