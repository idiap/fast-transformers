//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;

__device__ void get_result(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    float_accessor kv,
    float_accessor result,
    const int n,
    const int h,
    const int e,
    const int m,
    const int L
) {
    for (int l=0; l<L; l++) {
        kv[n][h][e][m] += keys[n][h][l][e] * values[n][h][l][m];
        __syncthreads();
        float res = queries[n][h][l][e]*kv[n][h][e][m];
        atomicAdd(
            &result[n][h][l][m],
            res
        );
    }
}


__global__ void causal_dot_product_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    float_accessor kv,
    float_accessor result,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M,
    const int E_per_block,
    const int blocks_per_sequence,
    const int T,
    const int l_offset
) {

    const int sequence_index = blockIdx.x / blocks_per_sequence;
    int n = sequence_index / H;
    int h = sequence_index % H;

    int e_local = threadIdx.x / M;
    int e_start = ((blockIdx.x % blocks_per_sequence) * E_per_block);
    int e = e_start + e_local;
    int m = threadIdx.x % M;

    // Load the shared memory for KV
    const int shared_kv_size = E_per_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_results = shared_mem + shared_kv_size;
    float* shared_values = shared_results + M;
    float* shared_keys = shared_values + M*T;
    float* shared_queries = shared_keys + E_per_block*T;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.0;
    }

    int t_end = (T + l_offset) <= L ? T : L - l_offset;
    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int d = i % M;
        shared_values[i] = values[n][h][t][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_per_block); i += blockDim.x)
    {
        int t = int(i / E_per_block) + l_offset;
        int d = (i % E_per_block) + e_start;
        if (d < E) {
            shared_keys[i] = keys[n][h][t][d];
            shared_queries[i] = queries[n][h][t][d];
        }
    }
    __syncthreads();
    if ((n >= N) || (e >= E)) {
        return;
    }
    shared_kv[threadIdx.x] = kv[n][h][e][m];
    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        shared_kv[e_local*M + m] += shared_keys[t*E_per_block + e_local] * shared_values[t*M + m];
        __syncthreads();
        float res = shared_queries[t*E_per_block + e_local] * shared_kv[e_local*M + m];
        atomicAdd(
            &shared_results[m],
            res
        );
        __syncthreads();
        if (threadIdx.x < M) {
            float r1 = shared_results[threadIdx.x];
            atomicAdd(
                &result[n][h][l][m],
                r1
            );
            shared_results[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();
    kv[n][h][e][m] = shared_kv[e_local*M + m];
}

void causal_dot_product(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    torch::Tensor product
) {
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    auto kv = torch::zeros({N, H, E, M}, queries.options());

    int threads = 1024;

    // Shared mem max size is 48KB
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / M) *  M;
    threads = MUL_PER_BLOCK;
    const int blocks_per_sequence = ((E*M) + threads -1) / threads;

    const int E_per_block = MUL_PER_BLOCK / M;
    int blocks  = N*H*blocks_per_sequence;
    int shared_mem_const = (E_per_block + 1)*M;
    int shared_mem_per_time = (M + 2*E_per_block);
    const int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_forward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
     causal_dot_product_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_forward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_block, blocks_per_sequence, T, l_offset
        );
    }
}



// we need shared memory to store
// Forward direction
// keys, values, gradout
// kv, results
// Backward direction
// queries, gradout, values
// kv_backwards, results
// Shared memory usage
// Forward
// keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
// Backward
// queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
// Total memory:
__global__ void causal_dot_backward_query_key_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor grad_out,
    float_accessor kv,
    float_accessor kv_backwards,
    float_accessor grad_queries,
    float_accessor grad_keys,
    int N,
    int H,
    int L,
    int E,
    int M,
    const int M_per_block,
    const int blocks_per_sequence,
    const int T,
    const int l_offset
) {

    const int sequence_index = blockIdx.x / blocks_per_sequence;
    int n = sequence_index / H;
    int h = sequence_index % H;

    int m_local = threadIdx.x / E;
    int m_start = ((blockIdx.x % blocks_per_sequence)*M_per_block);
    int m = m_start + m_local;
    int e = threadIdx.x % E;

    // Load the shared memory
    // Forward memory
    // keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Backward memory
    // queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Load the shared memory for KV
    extern __shared__ float shared_mem[];
    const int shared_kv_size = M_per_block * E;
    float* shared_kv = shared_mem;
    float* shared_kv_bw = shared_mem + shared_kv_size;
    float* shared_results = shared_kv_bw + shared_kv_size;
    float* shared_results_bw = shared_results + E;
    float* shared_keys = shared_results_bw + E;
    float* shared_values = shared_keys + E*T;
    float* shared_gradout = shared_values + M_per_block*T;
    float* shared_queries_bw = shared_gradout + M_per_block*T;
    float* shared_values_bw = shared_queries_bw + E*T;
    float* shared_gradout_bw = shared_values_bw + M_per_block*T;

    if (threadIdx.x < E) {
        shared_results[threadIdx.x] = 0.0;
        shared_results_bw[threadIdx.x] = 0.0;
    }

    int t_end = (T + l_offset) <= L ? T : (L - l_offset);
    for (int i = threadIdx.x; i < (t_end*M_per_block); i += blockDim.x)
    {
        int t = int(i / M_per_block) + l_offset;
        int t_bw = L - t - 1;
        int d = (i % M_per_block) + m_start;
        if (d < M) {
            shared_values[i] = values[n][h][t][d];
            shared_gradout[i] = grad_out[n][h][t][d];
            shared_values_bw[i] = values[n][h][t_bw][d];
            shared_gradout_bw[i] = grad_out[n][h][t_bw][d];
        }
    }
    for (int i = threadIdx.x; i < (t_end*E); i += blockDim.x)
    {
        int t = int(i / E) + l_offset;
        int t_bw = L - t - 1;
        int d = (i % E);
        shared_keys[i] = keys[n][h][t][d];
        shared_queries_bw[i] = queries[n][h][t_bw][d];
    }
    __syncthreads();

    if ((n >= N) || (m >= M)) {
        return;
    }

    shared_kv[threadIdx.x] = kv[n][h][e][m];
    shared_kv_bw[threadIdx.x] = kv_backwards[n][h][e][m];

    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        shared_kv[m_local*E + e] += shared_keys[t*E + e] * shared_values[t*M_per_block + m_local];
        shared_kv_bw[m_local*E + e] += shared_queries_bw[t*E + e] * shared_gradout_bw[t*M_per_block + m_local];
        __syncthreads();
        float res = shared_gradout[t*M_per_block + m_local] * shared_kv[m_local*E + e];
        float res_bw = shared_values_bw[t*M_per_block + m_local] * shared_kv_bw[m_local*E + e];
        atomicAdd(
            &shared_results[e],
            res
        );
        atomicAdd(
            &shared_results_bw[e],
            res_bw
        );
        __syncthreads();
        if (threadIdx.x < E) {
            float rq = shared_results[threadIdx.x];
            float rk = shared_results_bw[threadIdx.x];
            atomicAdd(
                &grad_queries[n][h][l][e],
                rq
            );
            atomicAdd(
                &grad_keys[n][h][l_b][e],
                rk
            );
            shared_results[threadIdx.x] = 0.0;
            shared_results_bw[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();
    kv[n][h][e][m] = shared_kv[m_local*E + e];
    kv_backwards[n][h][e][m] = shared_kv_bw[m_local*E + e];
}


__global__ void causal_dot_backward_value_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor grad_out,
    float_accessor kv,
    float_accessor grad_keys,
    float_accessor grad_values,
    int N,
    int H,
    int L,
    int E,
    int M,
    int E_per_block,
    int blocks_per_sequence,
    int T,
    int l_offset
) {

    const int sequence_index = blockIdx.x / blocks_per_sequence;
    int n = sequence_index / H;
    int h = sequence_index % H;

    int e_local = threadIdx.x / M;
    int e_start = ((blockIdx.x % blocks_per_sequence) * E_per_block);
    int e = e_start + e_local;
    int m = threadIdx.x % M;

    // Load the shared memory for KV
    const int shared_kv_size = E_per_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_results = shared_mem + shared_kv_size;
    float* shared_gradout = shared_results + M;
    float* shared_keys = shared_gradout + M*T;
    float* shared_queries = shared_keys + E_per_block*T;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.0;
    }

    int t_end = (T + l_offset) <= L ? T : L - l_offset;
    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int t_bw = L - 1 - t;
        int d = i % M;
        shared_gradout[i] = grad_out[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_per_block); i += blockDim.x)
    {
        int t = int(i / E_per_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_per_block) + e_start;
        if (d < E) {
            shared_keys[i] = keys[n][h][t_bw][d];
            shared_queries[i] = queries[n][h][t_bw][d];
        }
    }
    __syncthreads();

    if ((n >= N) || (e >= E)){
        return;
    }

    shared_kv[threadIdx.x] = kv[n][h][e][m];
    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        shared_kv[e_local*M + m] += shared_queries[t*E_per_block + e_local] * shared_gradout[t*M + m];
        __syncthreads();
        float res = shared_keys[t*E_per_block + e_local] * shared_kv[e_local*M + m];
        atomicAdd(
            &shared_results[m],
            res
        );
        __syncthreads();
        if (threadIdx.x < M) {
            float r1 = shared_results[threadIdx.x];
            atomicAdd(
                &grad_values[n][h][l_b][m],
                r1
            );
            shared_results[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();
    kv[n][h][e][m] = shared_kv[e_local*M + m];
}


void causal_dot_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor grad_out,
    torch::Tensor grad_queries,
    torch::Tensor grad_keys,
    torch::Tensor grad_values
) {
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    auto kv = torch::zeros({N, H, E, M}, queries.options());
    auto kv_backward = torch::zeros({N, H, E, M}, queries.options());

    const int threads = 1024;
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / E) *  E;
    const int blocks_per_sequence = ((E*M) + MUL_PER_BLOCK -1) / MUL_PER_BLOCK;
    const int M_per_block = MUL_PER_BLOCK / E;
    int blocks  = N*H*blocks_per_sequence;

    // Forward memory
    // keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Backward memory
    // queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Total memory
    // 2*((E + 2*M_per_block)*T + (E+1)*M_per_block)
    int shared_mem_const = 2*E*(1+M_per_block);
    int shared_mem_per_time = 2*(E + 2*M_per_block);
    int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_qk_backward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);
    for (int l_offset=0; l_offset < L; l_offset += T) {
        causal_dot_backward_query_key_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_qk_backward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv_backward.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, M_per_block, blocks_per_sequence, T, l_offset
        );
    }

    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int blocks_per_sequence_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_block = MPB / M;
    const int blocks_value  = N*H*blocks_per_sequence_value;

    shared_mem_const = (E_per_block + 1)*M;
    shared_mem_per_time = (M + 2*E_per_block);
    T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_v_backward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);
    kv.zero_();
    for (int l_offset=0; l_offset < L; l_offset += T) {
        causal_dot_backward_value_kernel
            <<<blocks_value, MPB, shared_mem_v_backward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_block, blocks_per_sequence_value, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "causal_dot_backward",
        &causal_dot_backward,
        "Compute the gradients for the causal dot product."
    );
}
