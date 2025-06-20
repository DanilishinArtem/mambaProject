import torch.nn as nn
import torch

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim
        # SSM parameters for each head
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.C = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.D = nn.Linear(embed_dim, self.head_dim)
        self.W_out = nn.Linear(self.head_dim, embed_dim)
        self.head_norms = nn.ModuleList([nn.LayerNorm(self.head_dim) for _ in range(num_heads)])
        # Learnable weights and heads
        self.head_weights = nn.Parameter(torch.ones(num_heads))

        self.ffn1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ffn2 = nn.Linear(embed_dim * 4, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, emb):
        B, L, _ = emb.shape
        h = [torch.zeros(B, self.head_dim, device=emb.device) for _ in range(self.num_heads)]
        outputs = []
        for t in range(L):
            x_t = emb[:, t]  # (B, E)
            sum_heads = self.D(x_t)  # (B, head_dim)
            for i in range(self.num_heads):
                a_t = torch.tanh(self.W_a[i](x_t))           # (B, D)
                b_t = self.W_b[i](x_t)                       # (B, D)
                h[i] = a_t * h[i] + b_t 
                out = self.C[i](h[i]) * self.D(x_t)          # output projection (B, D)
                out = self.head_norms[i](out)                # normalize
                sum_heads += self.head_weights[i] * out      # weighted sum
            sum_heads = sum_heads / self.num_heads           # average for stability
            outputs.append(sum_heads.unsqueeze(1))           # (B, 1, D)
        z = torch.cat(outputs, dim=1)  # (B, L, head_dim)
        z = self.W_out(z)
        u = z + self.norm(z)
        ffn_out = self.act(self.ffn1(u))
        o = self.ffn2(ffn_out)
        h_out = u + o
        return self.proj(h_out)
    

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