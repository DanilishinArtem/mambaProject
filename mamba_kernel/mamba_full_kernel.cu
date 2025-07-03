// mamba_full_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mamba_full_kernel(
    const float* __restrict__ a_proj,    // (B, L, H, D)
    const float* __restrict__ b_proj,
    const float* __restrict__ w_proj,
    const float* __restrict__ C_weight,  // (dim, dim)
    const float* __restrict__ C_bias,    // (dim,)
    const float* __restrict__ head_weights, // (H,)
    float* __restrict__ out,             // (B, L, dim)
    int B, int L, int H, int D
) {
    int b_idx = blockIdx.x;
    int h_idx = threadIdx.y;
    int d_idx = threadIdx.x;

    if (b_idx >= B || h_idx >= H || d_idx >= D) return;

    int dim = H * D;
    float h_val = 0.0f;

    for (int t = 0; t < L; ++t) {
        int idx = ((b_idx * L + t) * H + h_idx) * D + d_idx;
        float a_val = a_proj[idx];
        float b_val = b_proj[idx];
        float w_val = w_proj[idx];

        h_val = a_val * h_val + b_val;

        // Compute c_out = C(h_flat)
        float c_val = 0.0f;
        for (int k = 0; k < dim; ++k) {
            int h_flat_idx = h_idx * D + d_idx;
            c_val += h_val * C_weight[h_flat_idx * dim + k];
        }
        int out_idx = ((b_idx * L + t) * H + h_idx) * D + d_idx;
        c_val += C_bias[h_idx * D + d_idx];

        float hw = head_weights[h_idx];
        out[out_idx] = hw * (c_val + w_val);
    }
}

__global__ void mamba_full_backward_kernel(
    const float* __restrict__ grad_out, // grad по выходу (B,L,H*D)
    const float* __restrict__ a_proj,   // (B,L,H,D)
    float* __restrict__ grad_a,         // (B,L,H,D)
    float* __restrict__ grad_b,         // (B,L,H,D)
    int B, int L, int H, int D
) {
    int b_idx = blockIdx.x;
    int h_idx = threadIdx.y;
    int d_idx = threadIdx.x;

    if (b_idx >= B || h_idx >= H || d_idx >= D) return;

    int dim = H * D;

    // grad_h accumulator, обратный проход по времени (RNN-like)
    float grad_h_next = 0.0f;

    for (int t = L - 1; t >= 0; --t) {
        int idx = ((b_idx * L + t) * H + h_idx) * D + d_idx;

        float grad_out_val = grad_out[idx];
        float a_val = a_proj[idx];

        // Градиент по b_t равен grad по h_t (накопленному из grad_out)
        grad_b[idx] = grad_out_val + grad_h_next * 0.0f;  // т.к. b_t влияет напрямую, без умножения

        // Градиент по a_t
        grad_a[idx] = grad_h_next * grad_out_val;

        // Обновляем grad_h_next для следующего шага (t-1)
        grad_h_next = grad_out_val * a_val + grad_h_next * a_val;
    }
}

at::Tensor mamba_full_forward(
    at::Tensor a_proj,
    at::Tensor b_proj,
    at::Tensor w_proj,
    at::Tensor C_weight,
    at::Tensor C_bias,
    at::Tensor head_weights
) {
    const auto B = a_proj.size(0);
    const auto L = a_proj.size(1);
    const auto H = a_proj.size(2);
    const auto D = a_proj.size(3);
    const auto dim = H * D;

    auto out = at::zeros({B, L, H, D}, a_proj.options());

    dim3 threads(D, H);
    dim3 blocks(B);

    mamba_full_kernel<<<blocks, threads>>>(
        a_proj.contiguous().data_ptr<float>(),
        b_proj.contiguous().data_ptr<float>(),
        w_proj.contiguous().data_ptr<float>(),
        C_weight.contiguous().data_ptr<float>(),
        C_bias.contiguous().data_ptr<float>(),
        head_weights.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        B, L, H, D
    );

    return out.view({B, L, dim});
}

std::vector<at::Tensor> mamba_full_backward(
    at::Tensor grad_out,
    at::Tensor a_proj,
    at::Tensor b_proj,
    int B, int L, int H, int D
) {
    auto grad_a = at::zeros_like(a_proj);
    auto grad_b = at::zeros_like(b_proj);

    dim3 threads(D, H);
    dim3 blocks(B);

    mamba_full_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        a_proj.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        B, L, H, D
    );

    return {grad_a, grad_b};
}