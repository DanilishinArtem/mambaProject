import torch
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_a = nn.Linear(dim, dim)  # Для delta
        self.W_b = nn.Linear(dim, dim)  # Для B
        self.W_out = nn.Linear(dim, dim)  # После selective_scan
        self.C = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and "head_weights" not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        nn.init.normal_(self.head_weights, mean=1.0, std=0.02)

    def forward(self, emb, padding_mask=None):
        B, L, D = emb.shape

        # Проекции
        delta = torch.tanh(self.W_a(emb))               # (B, L, D)
        B_proj = self.W_b(emb)                          # (B, L, D)

        # Selective Scan
        A = torch.ones_like(delta[:, :, :1])            # (B, L, 1)
        C_proj = torch.zeros_like(B_proj)               # (B, L, D)
        D_proj = torch.ones_like(B_proj[:, :, 0])       # (B, L)

        scan_out = selective_scan_fn(
            x=None,
            delta=delta,
            A=A,
            B=B_proj,
            C=C_proj,
            D=D_proj,
            z=None,
            delta_bias=None,
            delta_softplus=False,
        )  # (B, D, L)

        # Переводим обратно в (B, L, D)
        scan_out = scan_out.permute(0, 2, 1).contiguous()
        scan_out = self.W_out(scan_out)

        # Residual + Norm
        z = self.norm1(emb + self.dropout(scan_out))

        # FFN + Residual + Norm
        ffn_out = self.ffn(z)
        out = self.norm2(z + ffn_out)
        return out
    

class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Layers
        self.layers = nn.ModuleList([
            MambaPlusPlus_layer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.output_fc = nn.Linear(dim, vocab_size)
        
        # Initialize
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x):
        # Create padding mask
        padding_mask = (x == 0)
        
        # Embedding + positional encoding
        emb = self.embed(x)
        emb = emb + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(emb)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        # Output
        x = self.norm(x)
        logits = self.output_fc(x)
        return logits