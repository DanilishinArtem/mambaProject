import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CopyDataset, get_tokenizer, batch_covariance, attn_score_plot#, BayesianPrefixFilter
from config import Config

class CausalLinearCombination(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.dim = dim
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, bias=False)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.k - 1, 0))
        y = self.conv(x)
        return y.transpose(1, 2)
    


class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, exp_clip=20.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.exp_clip = exp_clip  # предел для экспоненты

        # Проекция входа на (x и Δ)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        # Depthwise conv (по временной оси)
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=d_conv, 
            groups=d_model, 
            padding=d_conv - 1
        )

        # Параметры SSM
        self.A = nn.Parameter(torch.empty(d_model, d_state))
        self.B = nn.Parameter(torch.empty(d_model, d_state))
        self.C = nn.Parameter(torch.empty(d_model, d_state))

        # Выходная проекция
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Лёгкая инициализация для стабильности
        nn.init.normal_(self.A, mean=-0.1, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.C, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity="relu")
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        B, L, D = x.shape

        # 1. Линейная проекция
        x_proj, delta = self.in_proj(x).chunk(2, dim=-1)

        # 2. Depthwise conv (по seq_len)
        x_proj = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)  # обратно в [B, L, D]

        # 3. SSM step-by-step
        state = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            exp_term = torch.clamp(self.A * delta[:, t].unsqueeze(-1), min=-self.exp_clip, max=self.exp_clip)
            state = state * torch.exp(exp_term) + self.B * x_proj[:, t].unsqueeze(-1)
            y = torch.sum(state * self.C, dim=-1)
            outputs.append(y)

            # Проверка на NaN
            if torch.isnan(y).any():
                print(f"[Warning] NaN detected at step {t}")
                break

        y = torch.stack(outputs, dim=1)  # [B, L, D]

        # 4. Выходная проекция
        return self.out_proj(y)


def ema(X, beta=0.001):
    # простой векторизованный EMA по префиксу
    B, L, D = X.shape
    out = torch.zeros_like(X)
    s = torch.zeros(B, D, device=X.device)
    for t in range(L):
        s = beta * s + (1 - beta) * X[:, t]
        out[:, t] = s / (1 - beta**(t+1))  # корректируем bias
    return out


class BayesianPrefixFilter(nn.Module):
    def __init__(self, dim, q=1e-3, r=0.5, decay=1.0, learnable=True, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.decay = float(decay)
        self.eps = float(eps)
        self.learnable = learnable

        if learnable:
            # store log-params so positivity is guaranteed via softplus/exp
            self.log_q = nn.Parameter(torch.full((dim,), float(torch.log(torch.tensor(q + 1e-12))), dtype=torch.float32))
            self.log_r = nn.Parameter(torch.full((dim,), float(torch.log(torch.tensor(r + 1e-12))), dtype=torch.float32))
        else:
            self.register_buffer("_q", torch.full((dim,), float(q)))
            self.register_buffer("_r", torch.full((dim,), float(r)))

    def forward(self, X, prior_mean0=None):
        # X: (B, L, D)
        B, L, D = X.shape
        device, dtype = X.device, X.dtype
        assert D == self.dim

        # get q and r in positive domain
        if self.learnable:
            # use softplus to get positive; softplus is smoother than exp for training stability
            q = F.softplus(self.log_q).view(1, D).to(device=device, dtype=dtype)
            r = F.softplus(self.log_r).view(1, D).to(device=device, dtype=dtype)
        else:
            q = self._q.view(1, D).to(device=device, dtype=dtype)
            r = self._r.view(1, D).to(device=device, dtype=dtype)

        # prior mean initial
        if prior_mean0 is None:
            prior_mean0 = torch.zeros(B, D, device=device, dtype=dtype)
        else:
            prior_mean0 = prior_mean0.to(device=device, dtype=dtype)

        # steady-state algebra to obtain alpha,beta (per-dim)
        disc = (q*q + 4*q*r).clamp_min(self.eps)           # (1,D)
        P_inf = (-q + torch.sqrt(disc)) / 2.0              # (1,D)
        prior_prime = P_inf + q                            # (1,D)
        K = prior_prime / (prior_prime + r)                # (1,D)

        alpha = (1.0 - K) * float(self.decay)              # (1,D)
        beta  = K                                          # (1,D)

        # vectorized geometric weighted sum:
        # compute alpha^k for k=0..L-1 -> shape (1,L,D)
        k_idx = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)  # (1,L,1)
        alpha_pows = alpha.view(1,1,D).pow(k_idx)    # (1,L,D)

        # scale X by alpha^{-k} = 1 / alpha^k (guard small alpha)
        # clamp alpha_pows minimum to avoid division by zero
        alpha_pows = alpha_pows.clamp_min(1e-30)
        inv_alpha_pows = 1.0 / alpha_pows           # (1,L,D)

        X_scaled = X * inv_alpha_pows               # (B,L,D)
        cumsum = X_scaled.cumsum(dim=1)             # (B,L,D)

        S = cumsum * beta.view(1,1,D)               # (B,L,D)

        # Pprod = alpha^(t+1) where t index corresponds to cumsum at same t
        Pprod = alpha.view(1,1,D).pow(k_idx + 1)    # (1,L,D)

        out = Pprod.squeeze(0).unsqueeze(0) * (prior_mean0.unsqueeze(1) + S)  # (B,L,D)
        return out
    

if __name__ == "__main__":
    tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(vocab_size=Config.vocab_size)
    dataset = CopyDataset(tokenizer, batch_size=1, vocab_size=Config.vocab_size, sequence_length=Config.sequence_length)
    embedding = nn.Embedding(Config.vocab_size, Config.d_model).cuda()


    # Different filters:
    # filter = CausalLinearCombination(dim=Config.d_model, k = 2).cuda()
    filter = BayesianPrefixFilter(Config.d_model).cuda()

    model = SimpleMambaBlock(d_model=Config.d_model, d_state=16).cuda()


    for item, batch in enumerate(dataset):
        x = batch['input_ids'][:,:-1].cuda()
        # y = batch['input_ids'][:,1:].cuda()
        # cov_y = batch_covariance(embedding(y).float(), by_time=False)
        # attn_score_plot(cov_y.detach().cpu(), path=Config.path, name="y_")


        hidden_state = embedding(x).float()
        cov_before = batch_covariance(hidden_state, by_time=False)
        attn_score_plot(cov_before.detach().cpu(), path=Config.path, name="before_")

        hidden_state = filter(hidden_state)
        hidden_state = model(hidden_state)

        cov_after = batch_covariance(hidden_state, by_time=False)
        attn_score_plot(cov_after.detach().cpu(), path=Config.path, name="after_")
        # if item == Config.steps:
        #     break