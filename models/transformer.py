import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=embed_dim*4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        emb = self.embed(x)            # (B, L, E)
        emb = emb.permute(1,0,2)       # (L, B, E) для Transformer
        out = self.encoder(emb)        # (L, B, E)
        out = out.permute(1,0,2)       # (B, L, E)
        logits = self.fc(out)          # (B, L, V)
        return logits