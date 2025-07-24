import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

class HarmonicMambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=4,
        headdim=128,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        n_oscillators=8,
        base_freq=0.01,
        learnable_init_states=True,
        bias=False,
        conv_bias=True,
        chunk_size=128,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.chunk_size = chunk_size
        # Основные параметры
        self.d_model = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * dim
        self.headdim = headdim
        assert self.d_inner % headdim == 0, "d_inner должен делиться на headdim"
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.n_oscillators = n_oscillators
        self.base_freq = base_freq
        
        # Проекции
        self.conv_dim = self.d_inner + 2 * ngroups * d_state
        d_in = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads + 2 * n_oscillators
        self.in_proj = nn.Linear(dim, d_in, bias=bias, **factory_kwargs)
        
        # Сверточный слой
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=d_conv, padding=d_conv-1,
            groups=self.conv_dim, bias=conv_bias,
            **factory_kwargs
        )
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='linear')
        if conv_bias:
            nn.init.zeros_(self.conv1d.bias)
        
        # Инициализация состояний
        if learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(1, self.nheads, self.headdim, d_state, **factory_kwargs)
            )
        
        # Параметры dt
        dt = torch.exp(
            torch.randn(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        # Осцилляторные системы
        self.frequencies = nn.Parameter(
            base_freq * torch.logspace(0, 1, n_oscillators, **factory_kwargs)
        )
        self.damping_factors = nn.Parameter(
            torch.linspace(0.5, 0.9, n_oscillators, **factory_kwargs)
        )
        
        # Адаптивные веса для осцилляторов
        self.oscillator_weights = nn.Parameter(
            torch.ones(n_oscillators, **factory_kwargs)
        )
        
        # Параметры основного SSM
        w0 = torch.empty(self.nheads, **factory_kwargs).uniform_(1.0, 16.0)
        self.omega_log = nn.Parameter(torch.log(w0))
        
        # Прямой путь
        self.D = nn.Parameter(torch.randn(self.nheads, **factory_kwargs) * 0.02)
        
        # Нормализация и выход
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias, **factory_kwargs)
        nn.init.kaiming_normal_(self.out_proj.weight)
        if bias:
            nn.init.zeros_(self.out_proj.bias)
        
        # Финальная адаптивная гамма
        self.gamma = nn.Parameter(torch.ones(1, **factory_kwargs))

    def harmonic_oscillator(self, t, freq, damping):
        """Решает уравнение гармонического осциллятора с затуханием"""
        omega = 2 * math.pi * freq
        decay = torch.exp(-damping * omega * t)
        phase = omega * t * torch.sqrt(1 - damping**2)
        return decay * torch.sin(phase)

    def forward(self, u, seq_idx=None):
        b, L, _ = u.shape
        
        # 1) Входные проекции
        zxbcdt = self.in_proj(u)
        z = zxbcdt[..., :self.d_inner]
        x_full = zxbcdt[..., self.d_inner:self.d_inner+self.conv_dim]
        dt = F.softplus(zxbcdt[..., self.d_inner+self.conv_dim:self.d_inner+self.conv_dim+self.nheads] + self.dt_bias)
        osc_params = zxbcdt[..., self.d_inner+self.conv_dim+self.nheads:]
        
        # 2) Свертка
        xBC = x_full.transpose(1, 2)
        xBC = self.conv1d(xBC)[..., :L]
        xBC = xBC.transpose(1, 2)
        x, B, C = torch.split(
            xBC, 
            [self.d_inner, self.ngroups*self.d_state, self.ngroups*self.d_state], 
            dim=-1
        )
        
        # 3) Подготовка временной сетки для осцилляторов
        time_grid = torch.cumsum(dt.mean(dim=-1, keepdim=True), dim=1)  # [b, L, 1]
        time_grid = time_grid / (time_grid[:, -1:] + 1e-7)  # Нормализация
        
        # 4) Генерация когерентных резонансных состояний (CRS)
        crs_output = torch.zeros(b, L, self.n_oscillators, device=u.device)
        
        for i in range(self.n_oscillators):
            freq = self.frequencies[i]
            damping = self.damping_factors[i]
            osc_signal = self.harmonic_oscillator(time_grid, freq, damping)
            crs_output[..., i] = osc_signal.squeeze(-1)
        
        # 5) Адаптивная комбинация осцилляторов
        weights = F.softmax(self.oscillator_weights, dim=0)
        combined_osc = torch.einsum('blo,o->bl', crs_output, weights)
        
        # 6) Модуляция основного сигнала
        x_modulated = x * (1 + combined_osc.unsqueeze(-1))
        
        # 7) Основное сканирование Mamba
        initial_states = None
        if hasattr(self, 'init_states'):
            initial_states = self.init_states.expand(b, -1, -1, -1)
        
        # Используем базовую матрицу A без осцилляторной модуляции
        A = -torch.exp(self.omega_log)
        
        y = mamba_chunk_scan_combined(
            rearrange(x_modulated, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        
        # 8) Комбинация с резонансными состояниями
        y = y + self.gamma * combined_osc.unsqueeze(-1) * z
        
        # 9) Выход
        y = self.norm(y)
        return self.out_proj(y)



class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            HarmonicMambaLayer(dim) for _ in range(num_layers)
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