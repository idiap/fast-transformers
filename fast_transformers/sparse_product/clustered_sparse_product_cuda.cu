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
typedef torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> int32_accessor_2d;
typedef torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> int32_accessor_1d;

#define EPB 16
#define QPB 16
#define KPB 16


/*
   The idea is to follow GEMM like CUDA implementation.
   We assume the Query matrix is re-arranged such that
   first block of queries corresponds to the first cluster,
   next to the next set of clusters and so on.
   Each block only operates on single cluster.
   We implement the rest of the kernel similar to this
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
*/
__global__ void clustered_sparse_dot_product_kernel(
    const float_accessor_4d queries,
    const float_accessor_4d keys,
    const int32_accessor_4d topk,
    const int32_accessor_2d indx_maps,
    const int32_accessor_3d q_start_indx,
    const int32_accessor_3d q_end_indx,
    float_accessor_4d product
) {
    int E = queries.size(3);
    int K = topk.size(3);

    extern __shared__ float shared_mem[];
    float* shared_queries = shared_mem;
    float* shared_keys = shared_queries + (EPB*QPB);
    float* shared_topk = shared_keys + (EPB*KPB);

    int n = indx_maps[blockIdx.x][0];
    int h = indx_maps[blockIdx.x][1];
    int c = indx_maps[blockIdx.x][2];
    int l_end = q_end_indx[n][h][c];

    if ((threadIdx.x == 0)) {
        if ((threadIdx.y + (blockIdx.y * KPB)) < K) {
            shared_topk[threadIdx.y] = topk[n][h][c][threadIdx.y + (blockIdx.y * KPB)];
        }
        else {
            shared_topk[threadIdx.y] = -1;
        }
    }
    __syncthreads();

    float res = 0.0;
    int rq_indx = q_start_indx[n][h][c] + (indx_maps[blockIdx.x][3] * QPB)  + threadIdx.x;
    int rk_indx = shared_topk[threadIdx.y];
    int cq_indx = threadIdx.y;
    int ck_indx = threadIdx.x;
    for (int m=0; m<((E + EPB - 1)/EPB); m++) {
        cq_indx = m*EPB + threadIdx.y;
        ck_indx = m*EPB + threadIdx.x;
        if ((rq_indx < l_end) && (cq_indx < E)) {
            shared_queries[threadIdx.x + (QPB * threadIdx.y)] = queries[n][h][rq_indx][cq_indx];
        }
        else {
            shared_queries[threadIdx.x + (QPB * threadIdx.y)] = 0;
        }
        if ((rk_indx > -1) && (ck_indx) < E) {
            shared_keys[threadIdx.y + (KPB * threadIdx.x)] = keys[n][h][rk_indx][ck_indx];
        }
        else{
            shared_keys[threadIdx.y + (KPB * threadIdx.x)] = 0;
        }
        __syncthreads();
        for (int e=0; e<EPB; e++) {
            res += shared_queries[threadIdx.x + (EPB * e)] * shared_keys[threadIdx.y + (EPB * e)];
        }
        __syncthreads();
    }
    if ((rq_indx < l_end) && ((threadIdx.y + (blockIdx.y * KPB)) < K)) {
        product[n][h][rq_indx][threadIdx.y + (blockIdx.y * KPB)] = res;
    }
}


/*
   Since all of our kernels are implemented to work
   on a single cluster at a time.
   This simply creates a bunch of mappings for us
   to know which blockId operates on which cluster, sequenceid,
   head and the starting query id
*/
__global__ void create_maps_kernel(
    const int32_accessor_3d block_counts,
    const int32_accessor_3d block_counts_cumsum,
    const int n_block_per_query,
    int32_accessor_2d indx_maps
) {
    int N = block_counts.size(0);
    int H = block_counts.size(1);
    int C = block_counts.size(2);

    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_idx / (H*C);
    int h = (full_idx - n*H*C) / C;
    int c = full_idx % C;
    if (n >= N) {
        return;
    }
    int indx = block_counts_cumsum[n][h][c];
    int blocks = block_counts[n][h][c];
    indx -= blocks;
    for (int b=0; b<blocks; b++) {
        indx_maps[indx][0] = n;
        indx_maps[indx][1] = h;
        indx_maps[indx][2] = c;
        indx_maps[indx][3] = int(b / n_block_per_query);
        indx += 1;
    }
}


/*
   Sparse dot-product between Queries and Keys.
   The keys to multiplied are defined by the top-k
   matrix
*/
void clustered_sparse_dot_product(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor topk,
    const torch::Tensor q_start_indx,
    const torch::Tensor q_end_indx,
    const torch::Tensor block_counts,
    const torch::Tensor block_counts_cumsum,
    const int total_blocks,
    torch::Tensor indx_maps,
    torch::Tensor product
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(Q.device());

    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int C = topk.size(2);
    int S = K.size(2);

    int threads = 1024;
    int blocks = ((N*H*C) + threads - 1) / threads;
    create_maps_kernel<<<blocks, threads>>>(
        block_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        block_counts_cumsum.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        1,
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    dim3 dimBlock(QPB, KPB);
    dim3 dimGrid(total_blocks, (k + KPB - 1)/KPB);
    const int shared_mem = (((KPB + QPB) * EPB) + KPB) * sizeof(float);
    clustered_sparse_dot_product_kernel<<<dimGrid, dimBlock, shared_mem>>>(
        Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        q_start_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        q_end_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        product.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


/*
   Once again each block works for a single cluster
   Each thread sums over all the responsible keys (in chunks)
*/
__global__ void clustered_sparse_dot_queries_backward_kernel(
    const float_accessor_4d grad_out,
    const float_accessor_4d queries,
    const float_accessor_4d keys,
    const int32_accessor_4d topk,
    const int32_accessor_2d indx_maps,
    const int32_accessor_3d q_start_indx,
    const int32_accessor_3d q_end_indx,
    float_accessor_4d grad_q,
    float_accessor_4d grad_k
) {
    int E = grad_q.size(3);
    int K = topk.size(3);

    extern __shared__ float shared_mem[];
    float* shared_grad = shared_mem;
    float* shared_keys = shared_grad + (KPB*QPB);
    float* shared_queries = shared_keys + (EPB*KPB);
    float* shared_topk = shared_queries + (EPB*QPB);

    int n = indx_maps[blockIdx.x][0];
    int h = indx_maps[blockIdx.x][1];
    int c = indx_maps[blockIdx.x][2];
    int l_end = q_end_indx[n][h][c];

    // Load all the top indices for all keys
    int thread_id = threadIdx.x + (threadIdx.y * blockDim.x);
    for (int t=thread_id; t<K; t+=(blockDim.x*blockDim.y)) {
        shared_topk[t] = topk[n][h][c][t];
    }
    __syncthreads();

    float res = 0.0;
    int rq_indx = q_start_indx[n][h][c] + (indx_maps[blockIdx.x][3] * QPB)  + threadIdx.x;
    int e_indx = threadIdx.y + (blockIdx.y  * EPB);
    float res_k = 0.0;
    for (int kb=0; kb<((K + KPB - 1)/KPB); kb++) {
        if ((rq_indx < l_end) && (e_indx < E)) {
            shared_queries[threadIdx.x + (QPB * threadIdx.y)] = queries[n][h][rq_indx][e_indx];
        }
        else {
            shared_queries[threadIdx.x + (QPB * threadIdx.y)] = 0;
        }
        int rk_indx = (kb*KPB) + threadIdx.y;
        if ((rq_indx < l_end) && (rk_indx < K)) {
            shared_grad[threadIdx.x + (QPB * threadIdx.y)] = grad_out[n][h][rq_indx][rk_indx];
        }
        else {
            shared_grad[threadIdx.x + (QPB * threadIdx.y)] = 0;
        }
        rk_indx = kb*KPB + threadIdx.x;
        if ((rk_indx <  K) && (e_indx < E)){
            shared_keys[threadIdx.x + (KPB * threadIdx.y)] = keys[n][h][shared_topk[rk_indx]][e_indx];
        }
        else{
            shared_keys[threadIdx.x + (KPB * threadIdx.y)] = 0;
        }
        __syncthreads();
        for (int k=0; k<KPB; k++) {
            res += shared_grad[threadIdx.x + (QPB * k)] * shared_keys[k + (threadIdx.y * KPB)];
        }
        res_k = 0.0;
        if ((rk_indx < K) && (e_indx < E)) {
            for (int q=0; q<QPB; q++) {
                res_k += (shared_queries[q + (threadIdx.y * QPB)] * shared_grad[q + (threadIdx.x * QPB)]);
            }
            atomicAdd(&grad_k[n][h][shared_topk[rk_indx]][e_indx], res_k);
        }
        __syncthreads();
    }
    if ((rq_indx < l_end) && (e_indx < E)) {
        grad_q[n][h][rq_indx][e_indx] = res;
    }
}


/*
   Sparse dot product backward pass.
 */
void clustered_sparse_dot_backward(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor groups,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_Q,
    torch::Tensor grad_K,
    const torch::Tensor q_start_indx,
    const torch::Tensor q_end_indx,
    const torch::Tensor block_counts,
    const torch::Tensor block_counts_cumsum,
    const int total_blocks,
    torch::Tensor indx_maps
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(Q.device());

    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int C = topk.size(2);
    int k = topk.size(3);
    int S = K.size(2);

    int threads = 1024;
    int blocks = ((N*H*C) + threads - 1) / threads;
    create_maps_kernel<<<blocks, threads>>>(
        block_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        block_counts_cumsum.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        1,
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    dim3 dimBlock(QPB, EPB);
    dim3 dimGrid(total_blocks, (E + EPB - 1)/EPB);
    const int mem_grad = QPB * KPB;
    const int mem_keys = KPB * EPB;
    const int mem_queries = QPB * EPB;
    const int shared_mem = (mem_grad + mem_queries + mem_keys + k) * sizeof(float);
    clustered_sparse_dot_queries_backward_kernel<<<dimGrid, dimBlock, shared_mem>>>(
        grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        q_start_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        q_end_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        grad_Q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_K.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


__global__ void clustered_sparse_weighted_average_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d values,
    const int32_accessor_4d topk,
    const int32_accessor_2d indx_maps,
    const int32_accessor_3d q_start_indx,
    const int32_accessor_3d q_end_indx,
    float_accessor_4d output

) {
    int E = output.size(3);
    int K = topk.size(3);
    extern __shared__ float shared_mem[];
    float* shared_weights = shared_mem;
    float* shared_values = shared_weights + (KPB*QPB);
    float* shared_topk = shared_values + (EPB*KPB);

    int n = indx_maps[blockIdx.x][0];
    int h = indx_maps[blockIdx.x][1];
    int c = indx_maps[blockIdx.x][2];
    int l_end = q_end_indx[n][h][c];

    // Load all the top indices for all keys
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    for (int t=thread_id; t<K; t+=(blockDim.x*blockDim.y)) {
        shared_topk[t] = topk[n][h][c][t];
    }
    __syncthreads();

    float res = 0.0;
    int rq_indx = q_start_indx[n][h][c] + (indx_maps[blockIdx.x][3] * QPB)  + threadIdx.x;
    int e_indx = threadIdx.y + (blockIdx.y  * EPB);
    for (int kb=0; kb<((K + KPB - 1)/KPB); kb++) {
        int rk_indx = kb*KPB + threadIdx.y;
        if ((rq_indx < l_end) && (rk_indx < K)) {
            shared_weights[threadIdx.x + (QPB * threadIdx.y)] = weights[n][h][rq_indx][rk_indx];
        }
        else {
            shared_weights[threadIdx.x + (QPB * threadIdx.y)] = 0;
        }
        rk_indx = kb*KPB + threadIdx.x;
        if ((rk_indx <  K) && (e_indx < E)){
            shared_values[threadIdx.x + (KPB * threadIdx.y)] = values[n][h][shared_topk[rk_indx]][e_indx];
        }
        else{
            shared_values[threadIdx.x + (KPB * threadIdx.y)] = 0;
        }
        __syncthreads();
        for (int k=0; k<KPB; k++) {
            res += shared_weights[threadIdx.x + (QPB * k)] * shared_values[k + (threadIdx.y * KPB)];
        }
        __syncthreads();
    }
    if ((rq_indx < l_end) && (e_indx < E)) {
        output[n][h][rq_indx][e_indx] = res;
    }
}


/*
   Weighted average of the "values" with attention weight
   stored in the "weights". The values to be selected are
   defined by the "topk"
 */
void clustered_sparse_weighted_average(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor topk,
    torch::Tensor output,
    const torch::Tensor q_start_indx,
    const torch::Tensor q_end_indx,
    const torch::Tensor block_counts,
    const torch::Tensor block_counts_cumsum,
    const int total_blocks,
    torch::Tensor indx_maps
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(weights.device());

    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int C = topk.size(2);
    int k = topk.size(3);
    int S = values.size(2);
    int E = values.size(3);

    int threads = 1024;
    int blocks = ((N*H*C) + threads - 1) / threads;
    create_maps_kernel<<<blocks, threads>>>(
        block_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        block_counts_cumsum.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        1,
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    dim3 dimBlock(QPB, EPB);
    dim3 dimGrid(total_blocks, (E + EPB - 1)/EPB);
    const int mem_weights = QPB * KPB;
    const int mem_values = KPB * EPB;
    const int shared_mem = (mem_weights + mem_values + k) * sizeof(float);
    clustered_sparse_weighted_average_kernel<<<dimGrid, dimBlock, shared_mem>>>(
        weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        q_start_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        q_end_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


__global__ void clustered_sparse_weighted_average_backward_kernel(
    const float_accessor_4d weights,
    const float_accessor_4d grad_out,
    const int32_accessor_4d topk,
    const int32_accessor_2d indx_maps,
    const int32_accessor_3d q_start_indx,
    const int32_accessor_3d q_end_indx,
    const int q_per_block,
    float_accessor_4d grad_v
) {
    int E = grad_out.size(3);
    int K = topk.size(3);

    extern __shared__ float shared_mem[];
    float* shared_grad = shared_mem;
    float* shared_weights = shared_grad + (EPB*q_per_block);
    float* shared_topk = shared_weights + (KPB*q_per_block);

    int n = indx_maps[blockIdx.x][0];
    int h = indx_maps[blockIdx.x][1];
    int c = indx_maps[blockIdx.x][2];
    int l_end = q_end_indx[n][h][c];

    // Load all the top indices
    int thread_id = threadIdx.x + (threadIdx.y * blockDim.x);
    for (int t=thread_id; t<K; t+=(blockDim.x*blockDim.y)) {
        shared_topk[t] = topk[n][h][c][t];
    }
    int q_indx;
    int k_indx;
    int e_indx;
    int q_indx_local;
    int e_indx_local;
    int q_start = q_start_indx[n][h][c] + (indx_maps[blockIdx.x][3] * q_per_block);

    for (int t=thread_id; t<(EPB*q_per_block); t+=(blockDim.x*blockDim.y)) {
        q_indx_local = t / EPB;
        e_indx_local = t % EPB;
        q_indx = q_start + q_indx_local;
        e_indx = e_indx_local + (blockIdx.y * EPB);
        if ((q_indx < l_end) && (e_indx < E)) {
            shared_grad[(q_indx_local*EPB) + e_indx_local] = grad_out[n][h][q_indx][e_indx];
        }
        else {
            shared_grad[(q_indx_local*EPB) + e_indx_local] = 0;
        }
    }

    int k_indx_local;
    for (int t=thread_id; t<(KPB*q_per_block); t+=(blockDim.x*blockDim.y)) {
        q_indx_local = t / KPB;
        k_indx_local = t % KPB;
        q_indx = q_start + q_indx_local;
        k_indx = k_indx_local + (blockIdx.z * KPB);
        if ((q_indx < l_end) && (k_indx < K)) {
            shared_weights[(q_indx_local*KPB) + k_indx_local] = weights[n][h][q_indx][k_indx];
        }
        else {
            shared_weights[(q_indx_local*KPB) + k_indx_local] = 0;
        }
    }
    __syncthreads();

    float res = 0;
    int k_id = threadIdx.x + (blockIdx.z*KPB);
    e_indx = (blockIdx.y * EPB) + threadIdx.y;
    if ((k_id < K) && (e_indx < E)) {
        k_indx = shared_topk[k_id];
        for (int t=0; t<q_per_block; t++) {
            res += shared_grad[(t*EPB) + threadIdx.y] * shared_weights[(t*KPB) + threadIdx.x];
        }
        atomicAdd(&grad_v[n][h][k_indx][e_indx], res);
    }
}


/*
   Sparse weighted average backward pass
 */
void clustered_sparse_weighted_average_backward(
    const torch::Tensor weights,
    const torch::Tensor values,
    const torch::Tensor topk,
    const torch::Tensor grad_out,
    torch::Tensor grad_weights,
    torch::Tensor grad_values,
    const torch::Tensor q_start_indx,
    const torch::Tensor q_end_indx,
    const torch::Tensor block_counts,
    const torch::Tensor block_counts_cumsum,
    const int total_blocks,
    torch::Tensor indx_maps
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(weights.device());

    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);
    int C = topk.size(2);
    int S = values.size(2);

    int threads = 1024;
    int blocks = ((N*H*C) + threads - 1) / threads;
    create_maps_kernel<<<blocks, threads>>>(
        block_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        block_counts_cumsum.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        1,
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    dim3 dimBlock(QPB, KPB);
    dim3 dimGrid(total_blocks, (k + KPB - 1)/KPB);
    const int shared_mem = (((KPB + QPB) * EPB) + KPB) * sizeof(float);
    clustered_sparse_dot_product_kernel<<<dimGrid, dimBlock, shared_mem>>>(
        grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        topk.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        q_start_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        q_end_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    dim3 dimBlockV(KPB, EPB);
    dim3 dimGridV(total_blocks, (E + EPB - 1)/EPB, (k + KPB - 1)/KPB);
    const int shared_mem_v = (((KPB + EPB) * QPB) + k) * sizeof(float);
    clustered_sparse_weighted_average_backward_kernel
        <<<dimGridV, dimBlockV, shared_mem_v>>>(
            weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            topk.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
            indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            q_start_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
            q_end_indx.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
            QPB,
            grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
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
