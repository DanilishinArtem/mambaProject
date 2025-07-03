
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

using at::Tensor;

// CUDA kernel
__global__ void mamba_scan_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int B, int L, int H, int D
) {
    int b_idx = blockIdx.x;
    int h_idx = threadIdx.y;
    int d_idx = threadIdx.x;

    if (b_idx >= B || h_idx >= H || d_idx >= D) return;

    int index = ((b_idx * L + 0) * H + h_idx) * D + d_idx;

    float h_val = 0.0f;
    for (int t = 0; t < L; ++t) {
        index = ((b_idx * L + t) * H + h_idx) * D + d_idx;
        float a_val = a[index];
        float b_val = b[index];
        h_val = a_val * h_val + b_val;
        out[index] = h_val;
    }
}

// Host function
Tensor mamba_scan_forward(Tensor a, Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be CUDA");

    const auto B = a.size(0);
    const auto L = a.size(1);
    const auto H = a.size(2);
    const auto D = a.size(3);

    Tensor out = torch::zeros_like(a);

    const int threads_x = D;        // dim3.x = D
    const int threads_y = H;        // dim3.y = H

    dim3 threads(threads_x, threads_y);      // (D, H)
    dim3 blocks(B);                          // (B)

    // Flatten tensors
    const float* a_ptr = a.contiguous().data_ptr<float>();
    const float* b_ptr = b.contiguous().data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    mamba_scan_kernel<<<blocks, threads>>>(
        a_ptr, b_ptr, out_ptr, B, L, H, D
    );

    return out;
}

// Optional: dummy backward (can be extended)
std::vector<Tensor> mamba_scan_backward(Tensor grad_h, Tensor a, Tensor b) {
    auto grad_a = torch::zeros_like(a);
    auto grad_b = grad_h.clone();  // ∂h/∂b = 1
    return {grad_a, grad_b};
}
