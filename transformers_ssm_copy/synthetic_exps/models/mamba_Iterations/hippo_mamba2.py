import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

def make_hippo_legendre(N, device=None, dtype=None):
    M = torch.zeros(N, N, device=device, dtype=dtype)
    for n in range(N):
        for k in range(N):
            if n > k:
                continue
            val = (2 * n + 1) * (-1.0) ** (n + k) * math.comb(k, n)
            M[n, k] = val
    M = -M
    return M

class MambaPlusPlus_layer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Основные размеры
        self.d_model = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * dim
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        # Параметры интегратора
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # In-projection: z (d_inner), x (d_inner), B/C (2 * ngroups * d_state), dt (nheads)
        self.conv_dim = self.d_inner + 2 * ngroups * d_state
        d_in = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(dim, d_in, bias=bias, **factory_kwargs)

        # Глубинная свёртка: сохранит длину сигнала (padding=d_conv-1)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.conv_dim,
            bias=conv_bias,
            **factory_kwargs,
        )

        # Начальные состояния для группы сканов (опционально)
        if learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, headdim, d_state, **factory_kwargs)
            )

        # dt_bias: инициализация через inverse softplus
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # HiPPO / Legendre Memory Units initialization
        # Вместо случайной A_log, генерируем A из HiPPO Legendre SSM
        # Собственные значения HiPPO
        hippo_A = make_hippo_legendre(self.d_state, device=device, dtype=dtype)
        eigs = torch.linalg.eigvals(hippo_A).real  # только действительные части

        # Берём первые nheads значений (или циклически)
        eigs = eigs[torch.arange(self.nheads) % self.d_state]
        eigs = torch.clamp(eigs, min=1e-4)
        self.A_log = nn.Parameter(eigs.abs().log())
        # D-параметр оставляем как skip-коэффициент
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))

        # Нормировка и выход
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        b, L, _ = u.shape

        # 1) Входная проекция
        zxbcdt = self.in_proj(u)  # (B, L, d_in)

        # 2) Распаковываем
        z = zxbcdt[..., :self.d_inner]
        x_full = zxbcdt[..., self.d_inner : self.d_inner + self.conv_dim]
        dt = F.softplus(zxbcdt[..., -self.nheads:] + self.dt_bias)

        # 3) Свёртка по x_full (C, L)
        xBC = x_full.transpose(1, 2)  # (B, conv_dim, L)
        xBC = self.conv1d(xBC)[..., :L]  # отбрасываем лишние символы
        xBC = xBC.transpose(1, 2)  # (B, L, conv_dim)

        # 4) Сплит на ветви: x, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # 5) Python-реализация SSM-скана
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            -torch.exp(self.A_log),
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=(
                repeat(self.init_states, "... -> b ...", b=b)
                if self.learnable_init_states
                else None
            ),
            dt_limit=self.dt_limit,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # 6) Нормировка и выход
        y = self.norm(y, z)
        return self.out_proj(y)



class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            MambaPlusPlus_layer(dim) for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            is_rms_norm=isinstance(self.norm_f, RMSNorm)
        )
        logits = self.lm_head(hidden_states)
        return {"logits": logits}