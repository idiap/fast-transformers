//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#include <limits>

#include <torch/extension.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float4_accessor;
typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float2_accessor;
typedef torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> long_accessor;


inline int ceildiv(int a, int b) {
    return (a + b - 1)/b;
}


__global__ void copy_masked(
    float4_accessor buffer,
    float2_accessor attn_mask,
    long_accessor key_lengths,
    float4_accessor output,
    int local_context,
    int l_start,
    int s_start
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int n = idx / buffer.stride(0);
    idx = idx - n*buffer.stride(0);
    int h = idx / buffer.stride(1);
    idx = idx - h*buffer.stride(1);
    int l_offset = idx / buffer.stride(2);
    idx = idx - l_offset*buffer.stride(2);
    int s_offset = idx;

    if (n >= buffer.size(0)) {
        return;
    }

    int l = l_start + l_offset;
    int s = s_start + s_offset;
    int k = s - l + local_context/2;

    if (k < 0 || k >= local_context) {
        return;
    }

    output[n][h][l][k] = buffer[n][h][l_offset][s_offset] + attn_mask[l][s];
}


__global__ void local_copy_scaled(
    float4_accessor factors,
    float4_accessor values,
    float4_accessor output
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int E = values.size(3);
    int n = idx / factors.stride(0) / E;
    idx = idx - n*factors.stride(0)*E;
    int h = idx / factors.stride(1) / E;
    idx = idx - h*factors.stride(1)*E;
    int l = idx / factors.stride(2) / E;
    idx = idx - l*factors.stride(2)*E;
    int k = idx / E;
    idx = idx - k*E;
    int e = idx;

    if (n >= values.size(0)) {
        return;
    }
    int local_context = factors.size(3);
    if (k < 0 || k >= local_context) {
        return;
    }

    int s = k + l - local_context/2;
    if (s < 0 || s >= values.size(2)) {
        return;
    }

    atomicAdd(
        &output[n][h][l][e],
        factors[n][h][l][k] * values[n][h][s][e]
    );
}


torch::Tensor local_dot_product(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor attn_mask,
    const torch::Tensor key_lengths,
    int local_context
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int S = keys.size(2);
    int E = queries.size(3);

    const int blocks = 64;

    // Allocate space for the output
    auto output = queries.new_full(
        {N, H, L, local_context},
        -std::numeric_limits<float>::infinity()
    );
    auto buffer = queries.new_zeros({N, H, blocks, blocks+local_context});

    const int threads = 1024;
    int cuda_blocks = ceildiv(buffer.numel(), threads);

    for (int l=0; l<L; l+=blocks) {
        int s_start = std::max(0, l-local_context/2);
        int s_end = std::min(S, l-local_context/2+local_context+blocks);
        int n_keys = s_end-s_start;
        int n_queries = std::min(L-l, blocks);
        auto buff = buffer.narrow(3, 0, n_keys).narrow(2, 0, n_queries);
        at::matmul_out(
            buff,
            queries.narrow(2, l, n_queries),
            keys.narrow(2, s_start, n_keys).transpose(2, 3)
        );
        copy_masked<<<cuda_blocks, threads>>>(
            buff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            attn_mask.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            key_lengths.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            local_context,
            l,
            s_start
        );
    }

    return output;
}


torch::Tensor local_weighted_average(
    const torch::Tensor attention,
    const torch::Tensor values
) {
    // Allocate space for the output
    auto output = torch::zeros_like(values);

    const int threads = 1024;
    int blocks = ceildiv(attention.numel()*values.size(3), threads);

    local_copy_scaled<<<blocks, threads>>>(
        attention.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "local_dot_product",
        &local_dot_product,
        "Compute the product of Q and K for a small context around each Q"
    );
    m.def(
        "local_weighted_average",
        &local_weighted_average,
        "Perform the weighted average of V for a small context around each Q"
    );
}
