//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <cooperative_groups.h>
#include <torch/extension.h>

using namespace cooperative_groups;

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;
typedef torch::PackedTensorAccessor32<int64_t, 4, torch::RestrictPtrTraits> int64_accessor_4d;


inline __device__ float dot(const float *a, const float *b, int n) {
    float s = 0;
    for (int i=0; i<n; i++) {
        s += (*a) * (*b);
        a++;
        b++;
    }
    return s;
}


inline __device__ void add_scaled(float *a, const float *b, float s, int n) {
    for (int i=0; i<n; i++) {
        atomicAdd(a, s * (*b));
        a++;
        b++;
    }
}


__global__ void sparse_dot_product_kernel(
    const float_accessor_4d queries,
    const float_accessor_4d keys,
    const int64_accessor_4d topk,
    float_accessor_4d products,
    int q_load
) {
    const int N = queries.size(0);
    const int H = queries.size(1);
    const int L = queries.size(2);
    const int E = queries.size(3);
    const int S = keys.size(2);
    const int hl = H*L;
    extern __shared__ float shared_qs[];

    int full_indx = q_load*blockIdx.x + threadIdx.x;
    int n = full_indx / (hl);
    int h = (full_indx - n*hl) / L;
    int l = (full_indx - n*hl) % L;
    if ((threadIdx.x < q_load) && ((q_load*blockIdx.x + threadIdx.x) < (N*L*H))) {
        int q_indx = threadIdx.x;
        float *s_ptr = shared_qs + q_indx;
        for (int e=0; e<E; e++) {
            *s_ptr = queries[n][h][l][e];
            s_ptr += q_load;
        }
    }
    __syncthreads();

    int q_indx = threadIdx.x % q_load;
    int topk_idx = threadIdx.x / q_load;
    int q_processed = (blockIdx.x*q_load) + q_indx;
    int seq_idx = q_processed / (hl);
    int h_idx = (q_processed - seq_idx*hl)/L;
    int l_idx = (q_processed - seq_idx*hl)%L;

    if ((seq_idx >= N) || (l_idx >= L) || (h_idx >= H)) {
        return;
    }

    float s = 0;
    const float *q_cur = shared_qs + q_indx;
    int k_idx = topk[seq_idx][h_idx][l_idx][topk_idx];

    //#pragma unroll 8
    for (int e=0; e<E; e++) {
        s += (*q_cur) * keys[seq_idx][h_idx][k_idx][e];
        q_cur += q_load;
    }
    products[seq_idx][h_idx][l_idx][topk_idx] = s;
}


void sparse_dot_product(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor topk,
    torch::Tensor product
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);

    int max_threads = 1024;
    int q_max = (48 * 1024)/(4*E) < L ? (48 * 1024)/(4*E):L;

    int q_load = (max_threads/k) < q_max ? (max_threads/k):q_max;
    int threads = q_load * k;

    const int shared_mem_queries = q_load * E * sizeof(float);
    int total_products = L*N*H*k;
    int blocks = ceil(float(total_products)/(q_load * k));

    sparse_dot_product_kernel<<<blocks,
                                threads,
                                shared_mem_queries>>>(
        Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
        product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        q_load
    );
}


__global__ void sparse_dot_backward_kernel(
    const float_accessor_4d queries,
    const float_accessor_4d keys,
    const int64_accessor_4d topk,
    const float_accessor_4d grad_out,
    float_accessor_4d grad_q,
    float_accessor_4d grad_k
) {
    const int N = queries.size(0);
    const int H = queries.size(1);
    const int L = queries.size(2);
    const int E = queries.size(3);
    const int S = keys.size(2);
    const int k = topk.size(3);

    int full_index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_index / (H*L*k);
    int h = (full_index - n*H*L*k) / (L*k);
    int l = (full_index - n*H*L*k - h*L*k) / k;
    int j = full_index % k;

    if (n >= N) {
        return;
    }

    const int key_index = topk[n][h][l][j];
    const float grad = grad_out[n][h][l][j];
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_q[n][h][l][e], grad * keys[n][h][key_index][e]);
    }
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_k[n][h][key_index][e], grad * queries[n][h][l][e]);
    }
}


void sparse_dot_backward(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_Q,
    torch::Tensor grad_K
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);

    int threads = 1024;
    int blocks = (N*H*L*k + threads - 1) / threads;
    sparse_dot_backward_kernel<<<blocks, threads>>>(
        Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
        grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_K.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


__global__ void sparse_weighted_average_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d values,
    const int64_accessor_4d topk,
    float_accessor_4d output,
    int N,
    int H,
    int L,
    int E,
    int k,
    int n_dim_per_thread
) {
    extern __shared__ float shared_mem[];
    int block_idx = blockIdx.x;
    if ((block_idx > N*H*L)){
        return;
    }

    int n = (block_idx) / (H*L);
    int h = (block_idx - n*H*L) / (L);
    int l = block_idx  % L;


    if ((threadIdx.x < k)) {
        shared_mem[k*E + threadIdx.x] = weights[n][h][l][threadIdx.x];
        shared_mem[(k*(E+1)) +  threadIdx.x] = topk[n][h][l][threadIdx.x];
    }

    __syncthreads();

    if (threadIdx.x < k) {
        int n_threads_per_key  = E / n_dim_per_thread;
        int j = threadIdx.x / n_threads_per_key ;
        int d_start = (threadIdx.x - j*n_threads_per_key) * n_dim_per_thread;

        int key_idx = int(shared_mem[(k*(E+1)) + j]);
        const float s = shared_mem[k*E + j];

        for(int i=0; i<n_dim_per_thread; i++) {
            int cur_d = d_start + i;
            float v = values[n][h][key_idx][cur_d];
            shared_mem[j + (cur_d * k)] =  v * s;
        }
    }
    __syncthreads();

    if ((threadIdx.x < E)) {
        float sum = 0;
        int start = threadIdx.x*k;
        for (int i=start; i<start+k; i++) {
            sum = sum + shared_mem[i];
        }
        output[n][h][l][threadIdx.x] = sum;
    }
}

void sparse_weighted_average(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor topk,
    torch::Tensor output
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);


    auto weights_a = weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto values_a = values.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto topk_a = topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>();
    auto output_a = output.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    //float* output_p = output.data_ptr<float>();

    int max_threads = 1024;
    int n_dim_per_thread = E;
    // We need at least E threads for the final reduction
    int threads = ceil((E * k)/n_dim_per_thread) > E ? ceil((E * k)/n_dim_per_thread):E;
    int total_products = L*N*H*k;
    int blocks = ceil(float(total_products)/(k));
    const int shared_mem = (((k * E) + 2*k)* sizeof(float));
    sparse_weighted_average_kernel<<<blocks,
                                     threads,
                                     shared_mem>>>(
        weights_a,
        values_a,
        topk_a,
        output_a,
        N,
        H,
        L,
        E,
        k,
        n_dim_per_thread
    );
}


__global__ void sparse_weighted_average_backward_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d values,
    const int64_accessor_4d topk,
    const float_accessor_4d grad_out,
    float_accessor_4d grad_weights,
    float_accessor_4d grad_values,
    int N,
    int H,
    int L,
    int E,
    int k,
    int dim_per_thread
) {
    int full_index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_index / (H*L*k);
    int h = (full_index - n*H*L*k) / (L*k);
    int l = (full_index - n*H*L*k - h*L*k) / k;
    int j = full_index % k;

    if (n >= N) {
        return;
    }
    int key_idx = topk[n][h][l][j];
    int start_dim = threadIdx.y * dim_per_thread;
    int end_dim = start_dim + dim_per_thread;
    if (threadIdx.y == 0) {
        grad_weights[n][h][l][j] = dot(
            &values[n][h][key_idx][0],
            &grad_out[n][h][l][0],
            E
        );
    }
    float weight = weights[n][h][l][j];
    for (int e=start_dim; e<end_dim; e++) {
        atomicAdd(
            &grad_values[n][h][key_idx][e],
            weight * grad_out[n][h][l][e]
        );
    }
}


void sparse_weighted_average_backward(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_weights,
    torch::Tensor grad_values
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);

    auto weights_a = weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto values_a = values.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto topk_a = topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>();
    auto grad_out_a = grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto grad_weights_a = grad_weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto grad_values_a = grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    int threads_x = 256;
    int threads_y = 4;
    int dim_per_thread = E / threads_y;
    dim3 threads(threads_x, threads_y);
    int blocks = (N*H*L*k + threads_x - 1)/threads_x;
    sparse_weighted_average_backward_kernel<<<blocks, threads>>>(
        weights_a,
        values_a,
        topk_a,
        grad_out_a,
        grad_weights_a,
        grad_values_a,
        N,
        H,
        L,
        E,
        k,
        dim_per_thread
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sparse_dot_product",
        &sparse_dot_product,
        "Compute the dot product only in the positions specified by topk."
    );
    m.def(
        "sparse_dot_backward",
        &sparse_dot_backward,
        "Compute the gradients for the sparse dot product."
    );
    m.def(
        "sparse_weighted_average",
        &sparse_weighted_average,
        "Average the values using the sparse attention."
    );
    m.def(
        "sparse_weighted_average_backward",
        &sparse_weighted_average_backward,
        "Compute the gradients for the sparse weighted average."
    );
}

