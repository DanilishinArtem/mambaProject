import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

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
        bias=False,
        conv_bias=True,
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Base dims
        self.d_model = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * dim
        self.headdim = headdim
        assert self.d_inner % headdim == 0, "d_inner divisible by headdim"
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        # Integrator params
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.learnable_init_states = learnable_init_states

        # Projections
        self.conv_dim = self.d_inner + 2 * ngroups * d_state
        d_in = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(dim, d_in, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=d_conv, padding=d_conv-1,
            groups=self.conv_dim, bias=conv_bias,
            **factory_kwargs
        )

        # Initial states - FIXED DIMENSIONS
        if learnable_init_states:
            # CORRECTED: (nheads, headdim, d_state)
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, d_state, **factory_kwargs)
            )

        # dt bias
        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs)*(math.log(dt_max)-math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Oscillator params for 1st and 2nd scan
        w0 = torch.empty(self.nheads, **factory_kwargs).uniform_(1.0,16.0)
        self.omega_log = nn.Parameter(w0.log())
        # Direct path
        self.D = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))

        # Norm & out
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias, **factory_kwargs)
        
        # Additional components for stability
        self.activation = nn.SiLU()
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, u, seq_idx=None):
        b, L, _ = u.shape
        # 1) In proj
        zxbcdt = self.in_proj(u)
        z = zxbcdt[...,:self.d_inner]
        x_full = zxbcdt[...,self.d_inner:self.d_inner+self.conv_dim]
        dt = F.softplus(zxbcdt[...,-self.nheads:]+self.dt_bias)

        # 2) Conv
        xB = x_full.transpose(1,2)
        xB = self.conv1d(xB)[...,:L]
        xB = xB.transpose(1,2)
        x, B, C = torch.split(xB,[self.d_inner,self.ngroups*self.d_state,self.ngroups*self.d_state],dim=-1)

        # Prepare initial states
        initial_states = None
        if self.learnable_init_states:
            # CORRECTED: [b, nheads, headdim, d_state]
            initial_states = self.init_states.unsqueeze(0).expand(b, -1, -1, -1)

        # 3) First scan: acceleration -> velocity
        omega = torch.exp(self.omega_log)
        A1 = -omega**2  # second derivative
        
        # First scan
        v = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A1,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            dt_limit=self.dt_limit,
        )
        v = rearrange(v, "b l h p -> b l (h p)")
        
        # Activation and stabilization
        v = self.activation(v)
        v = v * self.skip_scale  # Learnable scaling
        
        # 4) Second scan: velocity -> position
        # Use integrator (A=0)
        A2 = torch.zeros(self.nheads, device=u.device)
        
        # Second scan
        pos = mamba_chunk_scan_combined(
            rearrange(v, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A2,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            dt_limit=self.dt_limit,
        )
        pos = rearrange(pos, "b l h p -> b l (h p)")
        
        # Combine with original input
        y = pos + x  # Residual connection
        
        # 5) Output
        out = self.norm(y, z)
        return self.out_proj(out)



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