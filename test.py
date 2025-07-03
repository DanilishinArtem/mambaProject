import torch
import mamba_scan

a = torch.rand(200, 80, 400, 16).cuda()
b = torch.rand(200, 80, 400, 16).cuda()
out = mamba_scan.forward(a, b)
print(out.shape)