import torch.nn as nn
import torch

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

    def forward(self, emb): # emb (B, L, E)
        B, L, _ = emb.shape
        H, D = self.num_heads, self.head_dim

        a_proj = torch.tanh(self.W_a(emb)).view(B, L, H, D)     # (B, L, H, D)
        b_proj = self.W_b(emb).view(B, L, H, D)                 # (B, L, H, D)
        w_proj = self.W_out(emb).view(B, L, H, D)               # (B, L, H, D)

        h = torch.zeros(B, H, D, device=emb.device)
        outputs = []

        for t in range(L):
            a_t = a_proj[:,t]   # (B, H, D)
            b_t = b_proj[:,t]   # (B, H, D)
            w_t = w_proj[:,t]   # (B, H, D)

            h = a_t * h + b_t
            c_out = self.C(h.view(B, -1)).view(B, H, D)

            out = self.head_weights.view(1, -1, 1) * (c_out + w_t)      # (B, H, D)
            concat = out.reshape(B, -1)                                 # (B, H * D)
            outputs.append(concat.unsqueeze(1))                         # (B, 1, H * D)

        z = torch.cat(outputs, dim=1)                                   # (B, L, H * D = hidden_dim)

        u = z + self.norm(z)
        ffn = self.act(self.ffn1(u))
        out = self.ffn2(ffn)
        return self.proj(u + out)                                       # (B, L, hidden_dim)
    

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