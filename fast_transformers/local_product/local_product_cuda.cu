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


template <int LB=32, int KB=32, int EB=32>
__global__ void local_copy_scaled(
    float4_accessor factors,
    float4_accessor values,
    float4_accessor output,
    dim3 strides
) {
    int idx = blockIdx.x;
    int n = idx / strides.x;
    idx -= n*strides.x;
    int h = idx / strides.y;
    idx -= h*strides.y;
    int lblock = idx / strides.z;
    idx -= lblock*strides.z;
    int eblock = idx;

    int local_context = factors.size(3);

    int l_local = threadIdx.x / EB;
    int e_local = threadIdx.x - l_local*EB;
    int l = lblock * LB + l_local;
    int e = eblock * EB + e_local;

    if (n > factors.size(0)) {
        return;
    }

    extern __shared__ float shared_mem[];
    float * s_factors = shared_mem;
    float * s_values = s_factors + LB*KB;

    for (int k=0; k<local_context; k+=KB) {
        // Load the data in shared mem
        int s1 = l - local_context/2 + k;
        int s2 = s1 + LB;
        int scurrent = s1 + e_local;
        if (l < factors.size(2) && k + e_local < local_context && scurrent >= 0 && scurrent < values.size(2)) {
            s_factors[l_local*KB + e_local] = factors[n][h][l][k + e_local];
        } else {
            s_factors[l_local*KB + e_local] = 0;
        }
        if (e < values.size(3) && s1 >=0 && s1 < values.size(2)) {
            s_values[l_local*EB + e_local] = values[n][h][s1][e];
        } else {
            s_values[l_local*EB + e_local] = 0;
        }
        if (e < values.size(3) && s2 >=0 && s2 < values.size(2)) {
            s_values[(l_local+LB)*EB + e_local] = values[n][h][s2][e];
        } else {
            s_values[(l_local+LB)*EB + e_local] = 0;
        }
        __syncthreads();

        // Do the dot product
        float result = 0;
        #pragma unroll
        for (int k_local=0; k_local<KB; k_local++) {
            result += s_factors[l_local*KB + k_local] * s_values[(k_local + l_local)*EB + e_local];
        }
        if (l < factors.size(2) && e < values.size(3)) {
            output[n][h][l][e] += result;
        }
        __syncthreads();
    }
}


template <int LB=32, int KB=32, int EB=32>
__global__ void local_copy_scaled_transpose(
    float4_accessor factors,
    float4_accessor values,
    float4_accessor output,
    dim3 strides
) {
    int idx = blockIdx.x;
    int n = idx / strides.x;
    idx -= n*strides.x;
    int h = idx / strides.y;
    idx -= h*strides.y;
    int sblock = idx / strides.z;
    idx -= sblock*strides.z;
    int eblock = idx;

    int local_context = factors.size(3);

    int s_local = threadIdx.x / EB;
    int e_local = threadIdx.x - s_local*EB;
    int s = sblock * LB + s_local;
    int e = eblock * EB + e_local;

    if (n > factors.size(0)) {
        return;
    }

    extern __shared__ float shared_mem[];
    float * s_factors = shared_mem;
    float * s_values = s_factors + LB*KB;

    for (int k=0; k<local_context; k+=KB) {
        // Load the data in shared mem
        int l = s - (local_context-1)/2 + k;
        int l2 = l + LB;
        // load the values
        if (l >= 0 && l < factors.size(2) && e < values.size(3)) {
            s_values[s_local*EB + e_local] = values[n][h][l][e];
        } else {
            s_values[s_local*EB + e_local] = 0;
        }
        if (l2 >= 0 && l2 < factors.size(2) && e < values.size(3)) {
            s_values[(s_local + LB)*EB + e_local] = values[n][h][l2][e];
        } else {
            s_values[(s_local + LB)*EB + e_local] = 0;
        }

        // load factors
        int lcurrent = l+e_local;
        int kcurrent = k+e_local;
        if (lcurrent >= 0 && lcurrent < factors.size(2) && kcurrent < local_context) {
            s_factors[s_local*KB + e_local] = factors[n][h][l+e_local][k+e_local];
        } else {
            s_factors[s_local*KB + e_local] = 0;
        }
        __syncthreads();

        // Do the dot product
        float result = 0;
        #pragma unroll
        for (int k_local=0; k_local<KB; k_local++) {
            result += s_values[(s_local+k_local)*EB + e_local] * s_factors[s_local*KB + k_local];
        }

        if (s < values.size(2) && e < values.size(3)) {
            output[n][h][s][e] += result;
        }
        __syncthreads();
    }
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


std::tuple<torch::Tensor, torch::Tensor> local_dot_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor key_lengths,
    const torch::Tensor grad,
    int local_context
) {
    // Extract some shapes
    int N = grad.size(0);
    int H = grad.size(1);
    int L = grad.size(2);
    int K = grad.size(3);
    int E = keys.size(3);

    // Allocate space for the output
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);

    const int threads = 32*32;
    int lblocks = ceildiv(L, 32);
    int eblocks = ceildiv(E, 32);
    int blocks = N * H * lblocks * eblocks;
    int shared_mem = 32*32 * 3 * sizeof(float);
    dim3 strides(
        H*lblocks*eblocks,
        lblocks*eblocks,
        eblocks
    );

    local_copy_scaled<<<blocks, threads, shared_mem>>>(
        grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
    );
    local_copy_scaled_transpose<<<blocks, threads, shared_mem>>>(
        grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
    );

    return std::make_tuple(grad_queries, grad_keys);
}


torch::Tensor local_weighted_average(
    const torch::Tensor attention,
    const torch::Tensor values
) {
    // Extract some shapes
    int N = attention.size(0);
    int H = attention.size(1);
    int L = attention.size(2);
    int K = attention.size(3);
    int E = values.size(3);

    // Allocate space for the output
    auto output = torch::zeros_like(values);

    const int threads = 32*32;
    int lblocks = ceildiv(L, 32);
    int eblocks = ceildiv(E, 32);
    int blocks = N * H * lblocks * eblocks;
    int shared_mem = 32*32 * 3 * sizeof(float);
    dim3 strides(
        H*lblocks*eblocks,
        lblocks*eblocks,
        eblocks
    );

    local_copy_scaled<<<blocks, threads, shared_mem>>>(
        attention.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
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
        "local_dot_backward",
        &local_dot_backward,
        "Compute the gradient of local_dot_product"
    );
    m.def(
        "local_weighted_average",
        &local_weighted_average,
        "Perform the weighted average of V for a small context around each Q"
    );
}
