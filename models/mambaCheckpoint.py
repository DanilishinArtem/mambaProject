class MambaPlusPlus_regular(nn.Module):
    def __init__(self, vocab_size, embed_dim, head_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads
        # SSM параметры для каждой головы
        self.W_b_out = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_a = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_b = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_out = nn.ModuleList([nn.Linear(embed_dim, head_dim) for _ in range(num_heads)])
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
                h[i] = a_t * h[i] + b_t * self.W_b_out[i](x_t) 
                out = self.C[i](h[i]) * self.W_out[i](x_t)   # output projection (B, D)
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