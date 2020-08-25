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
typedef torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> int64_accessor_3d;
typedef torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> int32_accessor_4d;
typedef torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> int32_accessor_3d;



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


__global__ void clustered_sparse_dot_product_kernel(
    const float *queries,
    const float *keys,
    const int64_t *topk,
    const int *sequence_length,
    const int* block_map,
    const int* query_map,
    const int64_t* sorted_group_idx,
    float *products,
    int N,
    int H,
    int L,
    int E,
    int k,
    int S,
    int C,
    int blocks_per_sequence,
    int queries_per_block
) {
    extern __shared__ float shared_mem[];
    float* shared_keys = shared_mem;
    float* shared_queries = shared_mem + k*E;
    // Getting the cluster id for the current block
    // block_map stores the information about the 
    // cluster index that current block needs to operate on
    int cluster_idx = block_map[blockIdx.x];
    int n = (blockIdx.x / blocks_per_sequence) / H;
    int h = (blockIdx.x / blocks_per_sequence) % H;
    if (cluster_idx == -1) {
        return; 
    }

    if ((threadIdx.x < k)) {
        // Load the keys into the shared memory
        int topk_idx = threadIdx.x;
        int k_idx = topk[n*H*C*k + h*C*k + cluster_idx*k + topk_idx];
        const float* k_ptr = keys + (n*H*S*E + h*S*E + k_idx*E);
        float* s_ptr = shared_keys + topk_idx; 
        for (int i=0; i<E; i++) {
            *s_ptr = *k_ptr;
            s_ptr += k;
            k_ptr++;
        }
    }

    if ((threadIdx.x >= k) && (threadIdx.x < (k + queries_per_block))) {
        // Load the queries into the shared memory
        int l_idx = query_map[blockIdx.x] + (threadIdx.x - k);
        // This condition ensures we only load the queries
        // for the right cluster
        if (l_idx < query_map[blockIdx.x + 1]) {
            int l = sorted_group_idx[n*H*L + h*L + l_idx];
            if (l < sequence_length[n]) {
                const float* q_ptr = queries + (n*H*L*E + h*L*E + l*E);
                float* s_ptr = shared_mem + threadIdx.x*E; 
                for (int i=0; i<E; i++) {
                    *s_ptr = *q_ptr;
                    s_ptr++;
                    q_ptr++;
                }
            }
        }
    }
    __syncthreads();
    int q_local_idx = threadIdx.x / k;
    int k_local_idx = threadIdx.x % k;
    // query_map stores the index of the starting query to be
    // processed by the current block
    int l_idx = query_map[blockIdx.x] + q_local_idx;
    if (l_idx >= query_map[blockIdx.x + 1]) {
        return;
    }
    int l = sorted_group_idx[n*H*L + h*L + l_idx];
    if (l >= sequence_length[n]) {
        return;
    }
    
    float s = 0;
    float* k_ptr = shared_keys + k_local_idx;
    float* q_ptr = shared_queries + q_local_idx*E;
    for (int i=0; i<E; i++) {
        s += (*k_ptr) * (*q_ptr);
        k_ptr += k;
        q_ptr++;
    }

    // printf("q_indx:%d, k_indx:%d, l:%d, dp:%f\n", q_local_idx, k_local_idx, l, s);
    products[n*H*L*k + h*L*k + l*k + k_local_idx] = s;
}

__global__ void create_maps_kernel(
    const int* cluster_counts,
    int* block_map,
    int* query_map,
    const int N,
    const int H,
    const int C,
    const int blocks_per_sequence,
    const int queries_per_block
) {
    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_idx / H;
    int h = full_idx % H;
    if (n >= N) {
        return;
    }
    int* block = block_map + (full_idx * blocks_per_sequence);
    int* q_block = query_map + (full_idx * blocks_per_sequence);
    const int* counts = cluster_counts + (full_idx * C);
    int idx = 0;
    int total_q_count = 0;

    for (int i=C-1; i>=0; i--) {
    //for (int i=0; i<C; i++) {
        int q_count = 0;
        while (q_count < counts[i] - queries_per_block) {
            block[idx] = i;
            q_block[idx] = total_q_count;
            total_q_count += queries_per_block;
            q_count += queries_per_block;
            idx++;
        }
        int left_over_queries = counts[i] - q_count;
        if (left_over_queries != 0) {
            block[idx] = i;
            q_block[idx] = total_q_count;
            total_q_count += left_over_queries;
            idx++;
        }
    }
}
// Each block loads operates on a single cluster
// We load the k-keys in the shared memory
// A maximum of 192 keys with 64 dimension can be loaded
// for a shared memory of 48KB
// Each block also loads the queries into the shared memory
// to compute dot products
// Each thread in a block computes one dot-product
// So the number of threads should be $K*N_queries$

void clustered_sparse_dot_product(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor topk,
    const torch::Tensor lengths,
    const torch::Tensor block_map,
    const torch::Tensor query_map,
    const torch::Tensor cluster_counts,
    const torch::Tensor sorted_group_indices,
    torch::Tensor product
) {
    
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int C = topk.size(2);
    int S = K.size(2);

    float* queries_p = Q.data_ptr<float>();
    float* keys_p = K.data_ptr<float>();
    int64_t* topk_p = topk.data_ptr<int64_t>();
    int* lengths_p = lengths.data_ptr<int>();
    float* product_p = product.data_ptr<float>();
    int* block_map_p = block_map.data_ptr<int>();
    int* query_map_p = query_map.data_ptr<int>();
    int* cluster_counts_p = cluster_counts.data_ptr<int>();
    int64_t* sorted_group_indices_p = sorted_group_indices.data_ptr<int64_t>();

    int max_threads = 1024;
    int queries_per_block = (max_threads / k) < L ? (max_threads / k):L;
    int threads = queries_per_block * k;
    int blocks_per_sequence = ((L*k)/threads) + C + 1;

    int blocks_map = ((N*H) + max_threads - 1)/max_threads;
    create_maps_kernel<<<blocks_map, max_threads>>>(
        cluster_counts_p,
        block_map_p,
        query_map_p,
        N,
        H,
        C,
        blocks_per_sequence,
        queries_per_block
    ); 

    const int blocks = blocks_per_sequence * N * H;
    const int shared_mem_queries = (queries_per_block + k) * E * sizeof(float);
    clustered_sparse_dot_product_kernel<<<blocks, threads,
                                          shared_mem_queries>>>(
        queries_p,
        keys_p,
        topk_p,
        lengths_p,
        block_map_p,
        query_map_p,
        sorted_group_indices_p,
        product_p,
        N,
        H,
        L,
        E,
        k,
        S,
        C,
        blocks_per_sequence,
        queries_per_block
    );
}


__global__ void clustered_sparse_dot_backward_kernel(
    const float_accessor_4d queries,
    const float_accessor_4d keys,
    const int32_accessor_3d groups,
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

    const int c = groups[n][h][l]; 
    if ((n >= N) || (c == -1)) {
        return;
    }

    const int key_index = topk[n][h][c][j];
    const float grad = grad_out[n][h][l][j];
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_q[n][h][l][e], grad * keys[n][h][key_index][e]);
    }
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_k[n][h][key_index][e], grad * queries[n][h][l][e]);
    }
}


void clustered_sparse_dot_backward(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor groups,
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
    
    clustered_sparse_dot_backward_kernel<<<blocks, threads>>>(
        Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        groups.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
        grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_K.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


__global__ void clustered_sparse_weighted_average_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d values,
    const int32_accessor_3d groups,
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
    int c = groups[n][h][l];
    if (c == -1) {
        return;
    }
    if ((threadIdx.x < k)) {
        shared_mem[k*E + threadIdx.x] = weights[n][h][l][threadIdx.x]; 
        shared_mem[(k*(E+1)) +  threadIdx.x] = topk[n][h][c][threadIdx.x]; 
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

void clustered_sparse_weighted_average(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor groups,
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
    auto groups_a = groups.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
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
    clustered_sparse_weighted_average_kernel<<<blocks, threads, shared_mem>>>(
        weights_a,
        values_a,
        groups_a,
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


__global__ void clustered_sparse_weighted_average_backward_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d values,
    const int32_accessor_3d groups,
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
    int c = groups[n][h][l]; 
    if ((n >= N) || (c == -1)) {
        return;
    }
    int key_idx = topk[n][h][c][j];
    int start_dim = threadIdx.y * dim_per_thread;
    int end_dim = start_dim + dim_per_thread;
    if (threadIdx.y == 0) {
        grad_weights[n][h][l][j] = dot(
            &values[n][h][key_idx][0],
            &grad_out[n][h][l][0],
            E
        );
    }
    add_scaled(
        &grad_values[n][h][key_idx][start_dim],
        &grad_out[n][h][l][start_dim],
        weights[n][h][l][j],
        dim_per_thread
    );
}


void clustered_sparse_weighted_average_backward(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor groups,
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
    auto groups_a = groups.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
    auto topk_a = topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>();
    auto grad_out_a = grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto grad_weights_a = grad_weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto grad_values_a = grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    int threads_x = 256;
    int threads_y = 4;
    int dim_per_thread = E / threads_y;
    dim3 threads(threads_x, threads_y);
    int blocks = (N*H*L*k + threads_x - 1)/threads_x;

    clustered_sparse_weighted_average_backward_kernel<<<blocks, threads>>>(
        weights_a,
        values_a,
        groups_a,
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
        "clustered_sparse_dot_product",
        &clustered_sparse_dot_product,
        "Compute the dot product only in the positions specified by topk."
    );
    m.def(
        "clustered_sparse_dot_backward",
        &clustered_sparse_dot_backward,
        "Compute the gradients for the sparse dot product."
    );
    m.def(
        "clustered_sparse_weighted_average",
        &clustered_sparse_weighted_average,
        "Average the values using the sparse attention."
    );
    m.def(
        "clustered_sparse_weighted_average_backward",
        &clustered_sparse_weighted_average_backward,
        "Compute the gradients for the sparse weighted average."
    );
}

