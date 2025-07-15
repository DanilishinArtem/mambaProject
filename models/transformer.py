import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=False  # (L, B, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        # Треугольная маска: 1 в позиции, куда можно смотреть, 0 — в запретные (будущие) токены
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)

    def forward(self, x):  # x: (B, L)
        B, L = x.shape
        device = x.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        emb = self.embed(x) + self.pos_embed(positions)  # (B, L, E)
        emb = emb.permute(1, 0, 2)  # (L, B, E)

        attn_mask = self.generate_causal_mask(L, device)  # (L, L)
        out = self.encoder(emb, mask=attn_mask)  # (L, B, E)
        out = out.permute(1, 0, 2)  # (B, L, E)
        logits = self.lm_head(out)  # (B, L, vocab_size)
        return logits