import torch.nn as nn
import torch

class MambaPlusPlus_layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.D = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.C = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])

        self.head_weights = nn.Parameter(torch.ones(num_heads))

        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, emb):
        B, L, _ = emb.shape
        h = [torch.zeros(B, self.head_dim, device=emb.device) for _ in range(self.num_heads)]
        head_outputs = []
        for t in range(L):
            head_outs_t = []
            x_t = emb[:, t]  # (B, E)
            for i in range(self.num_heads):
                a_t = torch.tanh(self.W_a[i](x_t))           # (B, D)
                b_t = self.W_b[i](x_t)                       # (B, D)
                h[i] = a_t * h[i] + b_t 
                head_out = self.C[i](h[i]) * self.D[i](x_t)
                head_outs_t.append(head_out)
            concat = torch.cat(head_outs_t, dim=-1)
            head_outputs.append(concat.unsqueeze(1))
        z = torch.cat(head_outputs, dim=1)  # (B, L, head_dim)
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