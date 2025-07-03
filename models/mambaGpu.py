import torch.nn as nn
import torch
import mamba_cuda  # твоё расширение с CUDA ядром

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_a = nn.Linear(embed_dim, hidden_dim)
        self.W_b = nn.Linear(embed_dim, hidden_dim)
        self.W_out = nn.Linear(embed_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.head_weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, emb):  # emb (B, L, E)
        B, L, _ = emb.shape
        H, D = self.num_heads, self.head_dim

        a_proj = torch.tanh(self.W_a(emb)).view(B, L, H, D)  # (B, L, H, D)
        b_proj = self.W_b(emb).view(B, L, H, D)              # (B, L, H, D)
        w_proj = self.W_out(emb).view(B, L, H, D)            # (B, L, H, D)

        # параметры линейного слоя C — веса и смещения
        C_weight = self.C.weight  # (hidden_dim, hidden_dim)
        C_bias = self.C.bias      # (hidden_dim,)

        # head_weights (H,) -> (1, H, 1) для удобства
        head_weights = self.head_weights.view(1, H, 1)

        # Вызов CUDA ядра
        # Предполагается, что оно возвращает (B, L, hidden_dim)
        z = mamba_cuda.full_forward(a_proj, b_proj, w_proj, head_weights, C_weight, C_bias)

        u = z + self.norm(z)
        ffn = self.act(self.ffn1(u))
        out = self.ffn2(ffn)
        return self.proj(u + out)  # (B, L, hidden_dim)
    

class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [MambaPlusPlus_layer(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output_fc(x)
        return logits