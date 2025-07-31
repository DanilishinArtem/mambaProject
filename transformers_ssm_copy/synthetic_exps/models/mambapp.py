import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block
from einops import rearrange, repeat
import math
import torch.nn.functional as F
import numpy as np
print("[INFO] HIMARK")

def estimate_operator_norm(W, num_iterations=5):
    # Power iteration to approximate the largest singular value
    v = torch.randn(W.shape[1], device=W.device)
    v = v / torch.norm(v)
    for _ in range(num_iterations):
        u = W @ v
        v = W.T @ u
        v = v / torch.norm(v)
    singular_value = torch.abs(torch.dot(u, W @ v)) / torch.norm(u)
    return singular_value

def operator_norm_regularizer(W, M, epsilon=1e-8):
    norm = estimate_operator_norm(W)
    scale = M / torch.clamp(norm, min=epsilon)
    return W * scale

def std_regularizer(W, M):
    std = torch.std(W, unbiased=False)
    return W * (M / torch.clamp(std, min=1e-8))



class MambaPlusPlus_layer(nn.Module):
    def __init__(self, dim, num_heads, d_state=None, d_conv=4, expand=2, dropout=0.0, M=1.0, epsilon=1e-5, T=512):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_state = d_state
        self.M = M  # Bound on matrix norms
        self.epsilon = epsilon  # Error tolerance
        self.T = T  # Sequence length
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(self.dim / 16)  # Match Mamba's default
        # Compute theoretical d_state if not provided
        if d_state is None:
            c, C = 1.0, 1.0  # Universal constants (can be tuned)
            d_state = int((self.M**2 / (c * (self.dim**(5/2)))) * (
                np.log(C * self.T * self.dim**2 / self.epsilon) + self.M**2 / np.sqrt(self.dim)
            ))
            self.d_state = max(d_state, 32)  # Ensure minimum state size for stability

        print("[INFO] d_state: {}".format(self.d_state))
        # Input projection to mimic W_q x_t and W_k x_s
        self.input_projection = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for u_s ≈ α(t-s)

        self.m_learnable = nn.Parameter(torch.tensor(float(np.log(np.expm1(M)))))  # Initialize such that softplus(m) ≈ M_init
        self.M_learnable = lambda: nn.functional.softplus(self.m_learnable)  # Ensure M > 0

        # Mamba block with optimized state dimension
        self.mamba = Mamba(
            d_model=dim,
            d_state=self.d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Initialize Mamba parameters using Gaussian quadrature
        self.initialize_mamba_params()

        # Layer normalization
        self.norm = RMSNorm(dim)
        # Regularization function (applied non-in-place)
        self.weight_regularizer = lambda x: torch.clamp(x, -M, M)

    def initialize_mamba_params(self):
        # Initialize A and x_proj using Gaussian quadrature over [-a, a], a = M^2 / (d * d^1/2)
        a = self.M**2 / (self.dim * self.dim**0.5)
        nodes, weights = self.gaussian_quadrature(self.d_state, -a, a)
        
        # Initialize A_log (diagonal A = -exp(A_log), shape (d_inner, d_state))
        with torch.no_grad():
            lambda_prime = torch.tensor(nodes, dtype=torch.float32, device=self.mamba.A_log.device)
            A = -torch.exp(self.alpha * lambda_prime)  # Shape: (d_state,)
            A = A.unsqueeze(0).repeat(self.d_inner, 1)  # Shape: (d_inner, d_state)
            self.mamba.A_log.copy_(torch.log(torch.abs(A)))  # A_log stores log(|A|)

        # Initialize x_proj to produce B and C resembling Gaussian quadrature weights
        with torch.no_grad():
            weights = torch.tensor(weights, dtype=torch.float32, device=self.mamba.x_proj.weight.device)
            weights = weights / weights.abs().max()  # Normalize for stability
            x_proj_weight = self.mamba.x_proj.weight  # Shape: (dt_rank + 2 * d_state, d_inner)
            b_weight = x_proj_weight[self.dt_rank:self.dt_rank + self.d_state, :]  # Shape: (d_state, d_inner)
            c_weight = x_proj_weight[self.dt_rank + self.d_state:, :]  # Shape: (d_state, d_inner)
            weights = weights.unsqueeze(1).repeat(1, self.d_inner)  # Shape: (d_state, d_inner)
            b_weight.copy_(weights)
            c_weight.copy_(weights)

    def gaussian_quadrature(self, n, a, b):
        # Generate n Gaussian quadrature nodes and weights over [a, b]
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = (b - a) / 2 * nodes + (b + a) / 2
        weights = (b - a) / 2 * weights
        return nodes, weights

    def forward(self, hidden_states, residual=None, padding_mask=None):
        # Apply input projection with regularized weights (non-in-place)
        regularized_weight = self.weight_regularizer(self.input_projection.weight)
        projected_states = nn.functional.linear(hidden_states, regularized_weight, self.input_projection.bias)
        modulated_states = projected_states * self.alpha

        # Apply Mamba layer to modulated states
        modulated_states, residual = layer_norm_fn(
            modulated_states,
            self.norm.weight,
            self.norm.bias,
            prenorm=True,
            residual=residual,
            is_rms_norm=True
        )
        mamba_out = self.mamba(modulated_states)

        return mamba_out, residual

    def estimate_operator_norm(self, W, num_iterations=5):
        # Power iteration to approximate the largest singular value
        v = torch.randn(W.shape[1], device=W.device)
        v = v / torch.norm(v)
        for _ in range(num_iterations):
            u = W @ v
            v = W.T @ u
            v = v / torch.norm(v)
        singular_value = torch.abs(torch.dot(u, W @ v)) / torch.norm(u)
        return singular_value

    def operator_norm_regularizer(self, W, epsilon=1e-8):
        M = self.M_learnable()  # Use learnable bound
        norm = self.estimate_operator_norm(W)
        scale = M / torch.clamp(norm, min=epsilon)
        return W * scale
    
    def apply_weight_regularization(self):
        # Apply regularization to weights after forward/backward pass (e.g., in training loop)
        with torch.no_grad():
            self.input_projection.weight.copy_(self.weight_regularizer(self.input_projection.weight))
            self.mamba.x_proj.weight.copy_(self.weight_regularizer(self.mamba.x_proj.weight))
            self.mamba.out_proj.weight.copy_(self.weight_regularizer(self.mamba.out_proj.weight))

            # self.input_projection.weight.copy_(self.operator_norm_regularizer(self.input_projection.weight))
            # self.mamba.x_proj.weight.copy_(self.operator_norm_regularizer(self.mamba.x_proj.weight))
            # self.mamba.out_proj.weight.copy_(self.operator_norm_regularizer(self.mamba.out_proj.weight))


class MambaPlusPlusML(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            MambaPlusPlus_layer(dim, num_heads, dropout=dropout) for _ in range(num_layers)
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
            hidden_states, residual = layer(hidden_states, residual)

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
        return {"logits" : logits}