//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> int64_accessor_3d;
typedef torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> int_accessor_3d;
typedef torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> int_accessor_1d;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;

__global__ void clustered_broadcast_kernel(
    const float_accessor_4d y,
    const float_accessor_3d f,
    const int_accessor_3d g,
    float_accessor_4d x,
    const int_accessor_1d sequence_length,
    const int_accessor_3d block_map,
    const int_accessor_3d x_map,
    const int64_accessor_3d sorted_g_idx,
    int blocks_per_sequence,
    int x_per_block
) {
    const int N = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int E = x.size(3);
    const int C = y.size(2);

    extern __shared__ float shared_mem[];
    // Getting the cluster id for the current block
    // block_map stores the information about the 
    // cluster index that current block needs to operate on
    int n = (blockIdx.x / blocks_per_sequence) / H;
    int h = (blockIdx.x / blocks_per_sequence) % H;
    int block_id = blockIdx.x % blocks_per_sequence;
    int cluster_idx = block_map[n][h][block_id];

    if (cluster_idx == -1) {
        return; 
    }
    if (threadIdx.x == 0) {
        shared_mem[E] = f[n][h][cluster_idx];
    }
    if (threadIdx.x < E) {
        // Load the clustered computation into the shared memory
        shared_mem[threadIdx.x] = y[n][h][cluster_idx][threadIdx.x];
    }
    __syncthreads();
    int l_idx = x_map[n][h][block_id] + threadIdx.x;
    if (l_idx >= x_map[n][h][block_id + 1]) {
        return;
    }
    int l = sorted_g_idx[n][h][l_idx];
    if (l >= sequence_length[n]) {
        return;
    }
    float factor = shared_mem[E];
    float *src = shared_mem;
    for (int e=0; e<E; e++) {
        x[n][h][l][e] = (*src) * factor;
        src++;
    }
}


__global__ void create_maps_kernel(
    const int_accessor_3d cluster_counts,
    int_accessor_3d block_map,
    int_accessor_3d query_map,
    const int blocks_per_sequence,
    const int queries_per_block
) {
    const int N = cluster_counts.size(0);
    const int H = cluster_counts.size(1);
    const int C = cluster_counts.size(2);

    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_idx / H;
    int h = full_idx % H;
    if (n >= N) {
        return;
    }
    int idx = 0;
    int total_q_count = 0;

    for (int i=C-1; i>=0; i--) {
        int q_count = 0;
        while (q_count < cluster_counts[n][h][i] - queries_per_block) {
            block_map[n][h][idx] = i;
            query_map[n][h][idx] = total_q_count;
            total_q_count += queries_per_block;
            q_count += queries_per_block;
            idx++;
        }
        int left_over_queries = cluster_counts[n][h][i] - q_count;
        if (left_over_queries != 0) {
            block_map[n][h][idx] = i;
            query_map[n][h][idx] = total_q_count;
            total_q_count += left_over_queries;
            idx++;
        }
    }
}


/**
 * Broadcast the aggregated results from tensor Y back to tensor X based on
 * group indices G multiplied by factors F.
 */
void clustered_broadcast(
    const torch::Tensor Y,
    const torch::Tensor G,
    const torch::Tensor F,
    torch::Tensor X,
    const torch::Tensor lengths,
    const torch::Tensor block_map,
    const torch::Tensor query_map,
    const torch::Tensor cluster_counts,
    const torch::Tensor sorted_group_indices
) {
    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);
    int C = Y.size(2);

    int_accessor_3d block_map_acc = block_map.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
    int_accessor_3d query_map_acc = query_map.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
    int_accessor_3d cluster_counts_acc = cluster_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
    int64_accessor_3d sgi_acc = sorted_group_indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>();

    const int max_threads = 1024;
    const int threads = (max_threads < L) ? max_threads:L;
    const int queries_per_block = threads;
    int blocks_per_sequence = (L/threads) + C + 1;
    int blocks_map = ((N*H) + max_threads - 1)/max_threads;
    create_maps_kernel<<<blocks_map, max_threads>>>(
        cluster_counts_acc,
        block_map_acc,
        query_map_acc,
        blocks_per_sequence,
        queries_per_block
    );
    const int blocks = blocks_per_sequence * N * H;
    const int shared_mem =  (E+1) * sizeof(float);

    clustered_broadcast_kernel<<<blocks, threads, shared_mem>>>(
        Y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        F.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        G.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        X.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        block_map_acc,
        query_map_acc,
        sgi_acc,
        blocks_per_sequence,
        queries_per_block
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "clustered_broadcast",
        &clustered_broadcast,
        "Broadcast the vectors of Y based on the indices "
        "in G multiplied by F back to X."
    );
}
