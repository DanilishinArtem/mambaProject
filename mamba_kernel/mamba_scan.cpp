#include <torch/extension.h>

torch::Tensor mamba_scan_forward(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> mamba_scan_backward(torch::Tensor grad_h, torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mamba_scan_forward, "MambaScan forward (CUDA)");
    m.def("backward", &mamba_scan_backward, "MambaScan backward (CUDA)");
}
