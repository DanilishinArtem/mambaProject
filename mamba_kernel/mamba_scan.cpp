#include <torch/extension.h>

at::Tensor mamba_full_forward(
    at::Tensor a_proj,
    at::Tensor b_proj,
    at::Tensor w_proj,
    at::Tensor C_weight,
    at::Tensor C_bias,
    at::Tensor head_weights
);

std::vector<at::Tensor> mamba_full_backward(
    at::Tensor grad_out,
    at::Tensor a_proj,
    at::Tensor b_proj,
    int B, int L, int H, int D
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("full_forward", &mamba_full_forward, "Full forward");
    m.def("full_backward", &mamba_full_backward, "Full backward");
}