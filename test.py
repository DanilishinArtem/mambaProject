import torch
import mamba_cuda

B, L, H, D = 2, 8, 4, 16
dim = H * D

a_proj = torch.rand(B, L, H, D, device='cuda', requires_grad=True)
b_proj = torch.rand(B, L, H, D, device='cuda', requires_grad=True)
w_proj = torch.rand(B, L, H, D, device='cuda')

C_weight = torch.randn(dim, dim, device='cuda')
C_bias = torch.randn(dim, device='cuda')
head_weights = torch.randn(H, device='cuda')

out = mamba_cuda.full_forward(a_proj, b_proj, w_proj, C_weight, C_bias, head_weights)

grad_out = torch.ones_like(out)

grad_a, grad_b = mamba_cuda.full_backward(grad_out, a_proj, b_proj, B, L, H, D)

print(grad_a.shape, grad_b.shape)
