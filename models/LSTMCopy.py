import torch.nn as nn

class LSTMCopy(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embed(x)  # (B, L, E)
        out, _ = self.lstm(emb)  # (B, L, H)
        logits = self.fc(out)    # (B, L, V)
        return logits