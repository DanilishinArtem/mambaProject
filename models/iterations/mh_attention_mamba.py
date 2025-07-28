import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block
import math

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.D, self.H = dim, num_heads

        # входная и выходная проекции
        self.in_proj   = nn.Linear(dim, dim)
        self.out_proj  = nn.Linear(dim, dim)
        self.post_norm = nn.LayerNorm(dim)

        # SSM-блоки по головам (каждый видит полный d_model)
        self.mamba_heads = nn.ModuleList([
            create_block(
                d_model=self.D,
                d_intermediate=0,
                ssm_cfg={"d_state": 32, "layer" : "Mamba2"},
                layer_idx=i
            ) for i in range(self.H)
        ])

        # self.mamba_heads = create_block(
        #     d_model=self.D,
        #     d_intermediate=0,
        #     ssm_cfg={"d_state": 32, "layer" : "Mamba2"},
        #     layer_idx=0
        # )

        # обучаемые смещения (в долях длины)
        self.head_shifts = nn.Parameter(torch.linspace(0, 1, self.H))

        # маршрутизатор для гейтирования голов по токенам
        self.router = nn.Sequential(
            nn.Linear(dim, self.H),
            nn.Softmax(dim=-1)
        )

        # кросс-хед внимание (по головам)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.D, num_heads=min(4, self.H), batch_first=True)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """
        x: (B, L, D)
        residual: (B, L, D) или None
        """
        B, L, D = x.shape
        if residual is None:
            residual = x

        # prenorm + проекция
        x = self.post_norm(x)
        x = self.in_proj(x)                      # (B, L, D)

        # вычисляем смещения в шагах
        shifts = (self.head_shifts * L).round().long()  # (H,)

        # получаем выходы голов (каждая full dim)
        heads = []
        for i, head in enumerate(self.mamba_heads):
            xi = torch.roll(x, shifts=shifts[i].item(), dims=1)
            yi, _ = head(xi)                        # (B, L, D)
            heads.append(yi)
        heads = torch.stack(heads, dim=2)         # (B, L, H, D)

        # for i in range(self.H):
        #     xi = torch.roll(x, shifts=shifts[i].item(), dims=1)
        #     yi, _ = self.mamba_heads(xi)                        # (B, L, D)
        #     heads.append(yi)
        # heads = torch.stack(heads, dim=2)         # (B, L, H, D)

        # кросс-хед внимание: каждая голова может «общаться»
        heads_flat = heads.view(B*L, self.H, D)    # (B·L, H, D)
        attn_out, _ = self.cross_attn(heads_flat, heads_flat, heads_flat)
        attn_out = attn_out.view(B, L, self.H, D)  # (B, L, H, D)

        # гейтирование по контексту
        gates = self.router(residual)              # (B, L, H)
        gates = gates.unsqueeze(-1)                # (B, L, H, 1)

        # объединяем: сумма по головам с учётом гейтов
        combined = (attn_out * gates).sum(dim=2) / math.sqrt(self.H)  # (B, L, D)

        # вывод final
        out = self.out_proj(combined)              # (B, L, D)
        out = self.post_norm(out)

        return out, residual


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