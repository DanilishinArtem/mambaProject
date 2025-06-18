import torch.nn as nn
import torch

class MambaPlusPlus_regular(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Parameter generators for each head
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_out = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.C   = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        # Projection and FFN
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act = nn.GELU()
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

# Standart mamba++
    def forward(self, x):
        # x: (B, L)
        emb = self.embed(x)  # (B, L, E)
        B, L, _ = emb.shape
        # Initialize hidden states for each head
        h = [torch.zeros(B, self.head_dim, device=x.device) for _ in range(self.num_heads)]
        head_outputs = []
        # SSM update per time step
        for t in range(L):
            head_outs_t = []
            for i in range(self.num_heads):
                a_t = torch.tanh(self.W_a[i](emb[:, t]))                             # gating
                b_t = self.W_b[i](emb[:, t])                                         # input transform
                h[i] = a_t * h[i] + b_t #* (emb[:, t])                                # SSM recurrence
                head_out = self.C[i](h[i]) * self.W_out[i](emb[:, t])                # output projection
                head_outs_t.append(head_out)
            # Concatenate heads
            concat = torch.cat(head_outs_t, dim=-1)       # (B, hidden_dim)
            head_outputs.append(concat.unsqueeze(1))
        z = torch.cat(head_outputs, dim=1)  # (B, L, hidden_dim)
        # Residual + Norm
        u = z + self.norm(z)
        # Feed-Forward Network
        ffn_out = self.act(self.ffn1(u))
        o = self.ffn2(ffn_out)
        h_out = u + o  # residual
        # Final output
        logits = self.output_fc(self.proj(h_out))
        return logits


class MambaPlusPlus_optimized(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Parameter generators for each head
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_out = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.C   = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        # Projection and FFN
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act = nn.GELU()
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

# Optimized mamba++
    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        E, H, D = self.embed.embedding_dim, self.num_heads, self.head_dim
        # Embed
        emb = self.embed(x)  # (B, L, E)
        # Stack parameters: W_a, W_b, W_out, C => shape (H, E, D) each
        # We precompute projections for all heads in parallel
        a_t = torch.stack([torch.tanh(W(emb)) for W in self.W_a], dim=2)     # (B, L, H, D)
        b_t = torch.stack([W(emb) for W in self.W_b], dim=2)                 # (B, L, H, D)
        wout = torch.stack([W(emb) for W in self.W_out], dim=2)              # (B, L, H, D)
        # Prepare for recurrent scan
        h = torch.zeros(B, H, D, device=x.device)  # hidden states per head
        outputs = []
        # Efficient scan over sequence (manually unrolled)
        for t in range(L):
            a = a_t[:, t]         # (B, H, D)
            b = b_t[:, t]         # (B, H, D)
            emb_t = emb[:, t].unsqueeze(1)  # (B, 1, E)
            emb_exp = emb_t.expand(-1, H, -1)  # (B, H, E)
            x_proj = emb_exp if E == D else b  # fallback
            h = a * h + b * x_proj  # (B, H, D)
            out = torch.stack([self.C[i](h[:, i]) for i in range(H)], dim=1)  # (B, H, D)
            out = out * wout[:, t]  # apply learned modulation
            outputs.append(out.unsqueeze(1))  # (B, 1, H, D)
        # Combine outputs
        z = torch.cat(outputs, dim=1)  # (B, L, H, D)
        z = z.view(B, L, H * D)        # (B, L, hidden_dim)
        # Residual + LayerNorm
        u = z + self.norm(z)
        # FFN
        ffn_out = self.act(self.ffn1(u))
        o = self.ffn2(ffn_out)
        h_out = u + o
        # Output
        logits = self.output_fc(self.proj(h_out))  # (B, L, vocab_size)
        return logits
    

class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [MambaPlusPlus_regular(embed_dim, embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # x: (B, L)
        x = self.embed(x)  # (B, L, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output_fc(x)  # (B, L, vocab_size)
        return logits