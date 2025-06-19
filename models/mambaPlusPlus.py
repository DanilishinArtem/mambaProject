import torch.nn as nn
import torch

# class MambaPlusPlus_regular(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
#         # Parameter generators for each head
#         self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.W_out = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.C   = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
#         # Projection and FFN
#         self.proj = nn.Linear(hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
#         self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
#         self.act = nn.GELU()
#         self.output_fc = nn.Linear(hidden_dim, vocab_size)

# # Standart mamba++
#     def forward(self, x):
#         # x: (B, L)
#         emb = self.embed(x)  # (B, L, E)
#         B, L, _ = emb.shape
#         # Initialize hidden states for each head
#         h = [torch.zeros(B, self.head_dim, device=x.device) for _ in range(self.num_heads)]
#         head_outputs = []
#         # SSM update per time step
#         for t in range(L):
#             head_outs_t = []
#             for i in range(self.num_heads):
#                 a_t = torch.tanh(self.W_a[i](emb[:, t]))                             # gating
#                 b_t = self.W_b[i](emb[:, t])                                         # input transform
#                 h[i] = a_t * h[i] + b_t #* (emb[:, t])                                # SSM recurrence
#                 head_out = self.C[i](h[i]) * self.W_out[i](emb[:, t])                # output projection
#                 head_outs_t.append(head_out)
#             # Concatenate heads
#             concat = torch.cat(head_outs_t, dim=-1)       # (B, hidden_dim)
#             head_outputs.append(concat.unsqueeze(1))
#         z = torch.cat(head_outputs, dim=1)  # (B, L, hidden_dim)
#         # Residual + Norm
#         u = z + self.norm(z)
#         # Feed-Forward Network
#         ffn_out = self.act(self.ffn1(u))
#         o = self.ffn2(ffn_out)
#         h_out = u + o  # residual
#         # Final output
#         logits = self.output_fc(self.proj(h_out))
#         return logits


class MambaPlusPlus_regular(nn.Module):
    def __init__(self, vocab_size, embed_dim, head_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads
        # SSM параметры для каждой головы
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.C = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(num_heads)])
        # Learnable веса на головы
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        # LayerNorm на каждую голову
        self.head_norms = nn.ModuleList([nn.LayerNorm(head_dim) for _ in range(num_heads)])
        # Дополнительный skip-путь Dx_t
        self.D = nn.Linear(embed_dim, head_dim)
        # FFN блок и выход
        self.ffn1 = nn.Linear(head_dim, head_dim * 4)
        self.ffn2 = nn.Linear(head_dim * 4, head_dim)
        self.proj = nn.Linear(head_dim, head_dim)
        self.output_fc = nn.Linear(head_dim, vocab_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(head_dim)

    def forward(self, x):
        emb = self.embed(x)  # (B, L, E)
        B, L, _ = emb.shape
        h = [torch.zeros(B, self.head_dim, device=x.device) for _ in range(self.num_heads)]

        outputs = []

        for t in range(L):
            x_t = emb[:, t]  # (B, E)
            sum_heads = self.D(x_t)  # (B, head_dim)

            for i in range(self.num_heads):
                a_t = torch.tanh(self.W_a[i](x_t))           # (B, D)
                b_t = self.W_b[i](x_t)                       # (B, D)
                h[i] = a_t * h[i] + b_t 
                out = self.C[i](h[i]) + self.D(x_t)          # output projection (B, D)
                out = self.head_norms[i](out)                # normalize
                sum_heads += self.head_weights[i] * out      # weighted sum

            sum_heads = sum_heads / self.num_heads           # average for stability
            outputs.append(sum_heads.unsqueeze(1))           # (B, 1, D)

        z = torch.cat(outputs, dim=1)  # (B, L, head_dim)
        # Residual + Norm
        u = z + self.norm(z)
        # Feed-Forward
        ffn_out = self.act(self.ffn1(u))
        o = self.ffn2(ffn_out)
        h_out = u + o
        logits = self.output_fc(self.proj(h_out))  # (B, L, vocab_size)
        return logits

# ------------------------------------------------------------

# class MambaPlusPlus_layer(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, num_heads):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
#         # Parameter generators for each head
#         self.W_a = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.W_b = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.W_out = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
#         self.C   = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
#         # Projection and FFN
#         self.proj = nn.Linear(hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
#         self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
#         self.act = nn.GELU()

#     def forward(self, emb):
#         B, L, _ = emb.shape
#         h = [torch.zeros(B, self.head_dim, device=emb.device) for _ in range(self.num_heads)]
#         head_outputs = []
#         # SSM update per time step
#         for t in range(L):
#             head_outs_t = []
#             for i in range(self.num_heads):
#                 a_t = torch.tanh(self.W_a[i](emb[:, t]))
#                 b_t = self.W_b[i](emb[:, t])
#                 h[i] = a_t * h[i] + b_t
#                 head_out = self.C[i](h[i]) * self.W_out[i](emb[:, t])
#                 head_outs_t.append(head_out)
#             # Concatenate heads
#             concat = torch.cat(head_outs_t, dim=-1)
#             head_outputs.append(concat.unsqueeze(1))
#         z = torch.cat(head_outputs, dim=1)
#         u = z + self.norm(z)
#         ffn_out = self.act(self.ffn1(u))
#         o = self.ffn2(ffn_out)
#         h_out = u + o
#         return self.proj(h_out)


class MambaPlusPlus_layer(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads
        # SSM параметры для каждой головы
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.C = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(num_heads)])
        # Learnable веса на головы
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        # LayerNorm на каждую голову
        self.head_norms = nn.ModuleList([nn.LayerNorm(head_dim) for _ in range(num_heads)])
        # Дополнительный skip-путь Dx_t
        self.D = nn.Linear(embed_dim, head_dim)
        # FFN блок и выход
        self.ffn1 = nn.Linear(head_dim, head_dim * 4)
        self.ffn2 = nn.Linear(head_dim * 4, head_dim)
        self.proj = nn.Linear(head_dim, head_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(head_dim)

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
                out = self.C[i](h[i]) + self.D(x_t)          # output projection (B, D)
                out = self.head_norms[i](out)                # normalize
                sum_heads += self.head_weights[i] * out      # weighted sum
            sum_heads = sum_heads / self.num_heads           # average for stability
            outputs.append(sum_heads.unsqueeze(1))           # (B, 1, D)
        z = torch.cat(outputs, dim=1)  # (B, L, head_dim)
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