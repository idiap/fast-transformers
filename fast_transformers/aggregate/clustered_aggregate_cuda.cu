//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Written by Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> int_accessor_1d;
typedef torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> int_accessor_2d;
typedef torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> int_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;

#define EPB 32
#define QPB 16
#define QPT 16

// We assume the queries have been rearranged such
// that first continuous block belongs to first cluster,
// next to second and so forth.

// The idea is that each thread now aggregates multiple
// Queries and then does one atomic write.
// If the cluster-id remains same we aggregate, else we
// do atomic write for that cluster and move-on.
// Threads in y dimension refer to the embedding dimension.

__global__ void clustered_aggregate_kernel(
    const float_accessor_4d x,
    const int_accessor_3d g,
    const float_accessor_3d f,
    float_accessor_4d y,
    const int_accessor_1d lengths,
    int Le
) {
    int N = x.size(0);
    int H = x.size(1);
    int L = x.size(2);
    int E = x.size(3);
    int C = y.size(2);

    int e_idx = threadIdx.y;

    int full_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int hl = H*Le;
    int n = full_idx / hl;
    int h = (full_idx % hl) / Le;
    int lb = full_idx - n*hl - h*Le;
    int l = lb*QPT;

    if ((n >= N)) {
        return;
    }
    int l_max = lengths[n];
    if (l >= l_max) {
        return;
    }
    int l_end = (l + QPT) < l_max ? (l + QPT):l_max;

    int k_cur = g[n][h][l];
    int k_prev = k_cur;
    for (int e=threadIdx.y; e<E; e+=EPB) {
        float res = 0.0;
        k_cur = g[n][h][l];
        k_prev = k_cur;
        float f_cur = f[n][h][k_cur];
        for (int i=l; i<l_end; i++) {
            k_cur = g[n][h][i];
            if (k_cur == k_prev) {
                res += (f_cur *  x[n][h][i][e]);
            }
            else {
                atomicAdd(&y[n][h][k_prev][e], res);
                f_cur = f[n][h][k_cur];
                k_prev = k_cur;
                res = (f_cur *  x[n][h][i][e]);
            }
        }
        atomicAdd(&y[n][h][k_cur][e], res);
    }
}


/**
 * Aggregate the passed vectors X based on group indices in G multiplied by
 * factors F.
 */
void clustered_aggregate(
    const torch::Tensor X,
    const torch::Tensor G,
    const torch::Tensor F,
    const torch::Tensor lengths,
    torch::Tensor Y
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(X.device());

    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);
    int C = Y.size(2);

    // Each thread works on QPT queries
    // Le = (L + QPT - 1) / QPT
    // There are QPB threads per block in x direction
    // blocks = ((N * H * Le) + QPB - 1) // QPB;
    int Le = (L + QPT - 1) / QPT;
    int blocks = ((N*H*Le) + QPB - 1) / QPB;

    dim3 dimBlock(QPB, EPB);
    clustered_aggregate_kernel<<<blocks, dimBlock>>>(
        X.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        G.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        F.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        Y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        Le
    );
}




// Each block works on a group of clusters
// The number of clusters in the group are
// decided by the size of shared memory
// Assume E dimension of each vector to be broadcasted
// #clusters * E < 192 * 64
// 192 * 64 are the total number of floats we can hold in
// shared memory
// The idea is the
// Originally we needed (N*H*L)/threads_size blocks
// To calculate the required number of blocks now
// we will additionally need the counts of clusters
// For group 1, we need N*H*(sum of counts in g1) / threads blocks
// For group 2, we need N*H*(sum of counts in g2) / threads blocks
// Note that in practice we will not need crazy number of groups.
// For E=64, a single group can hold 192 clusters.


__global__ void clustered_broadcast_kernel(
    const float_accessor_4d y,
    const int_accessor_3d sorted_g,
    const float_accessor_3d f,
    float_accessor_4d x,
    int_accessor_2d indx_maps,
    const int G
) {
    int N = x.size(0);
    int H = x.size(1);
    int L = x.size(2);
    int E = x.size(3);
    int C = y.size(2);

    extern __shared__ float shared_mem[];
    int n = indx_maps[blockIdx.x][0];
    int h = indx_maps[blockIdx.x][1];
    int g = indx_maps[blockIdx.x][2];
    int l = indx_maps[blockIdx.x][3] + threadIdx.x;
    int n_queries = indx_maps[blockIdx.x][4];

    // Load the values to broadcast and factors into shared memory
    int clusters_to_load = C / G;
    int cluster_offset = g * clusters_to_load;
    float* shared_values = shared_mem;
    float* shared_factors = shared_mem + clusters_to_load*E;
    for (int c=threadIdx.x; c<clusters_to_load; c+=blockDim.x) {
        for (int e=0; e<E; e++) {
            shared_values[e*clusters_to_load + c] = y[n][h][c+cluster_offset][e];
        }
        shared_factors[c] = f[n][h][c+cluster_offset];
    }
    __syncthreads();

    // Bail if out of bounds
    if (threadIdx.x >= n_queries) {
        return;
    }
    int k = sorted_g[n][h][l];
    // if ((k < 0) || (k >= C)) {
    //     return;
    // }
    k -= cluster_offset;
    // Copy-broadcast from y into x
    float factor = shared_factors[k];
    for (int e=0; e<E; e++) {
        x[n][h][l][e] = (shared_values[e*clusters_to_load + k] * factor);
    }
}

__global__
void create_maps(
    const int_accessor_3d group_counts,
    const int_accessor_3d block_counts,
    const int threads,
    int_accessor_2d indx_maps
) {
    if (threadIdx.x == 0) {
        int N = group_counts.size(0);
        int H = group_counts.size(1);
        int G = group_counts.size(2);
        int indx = 0;
        for (int n=0; n<N; n++){
            for (int h=0; h<H; h++) {
                int acc_g_count = 0;
                for (int g=0; g<G; g++) {

                    int q_id = acc_g_count;
                    int q_end_id = 0;
                    int g_count = group_counts[n][h][g];
                    int blocks = block_counts[n][h][g];
                    for (int b=0; b<blocks; b++) {
                        indx_maps[indx][0] = n;
                        indx_maps[indx][1] = h;
                        indx_maps[indx][2] = g;
                        indx_maps[indx][3] = q_id;
                        q_end_id += threads;
                        indx_maps[indx][4] = (q_end_id < g_count) ? threads:g_count - (b*threads);
                        q_id += threads;
                        indx++;
                    }
                    acc_g_count += g_count;
                }
            }
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
    const torch::Tensor block_counts,
    const torch::Tensor group_counts,
    const int threads,
    const int n_groups,
    const int blocks,
    torch::Tensor indx_maps
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(Y.device());

    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);
    int C = Y.size(2);
    create_maps<<<1, 1>>>(
        group_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        block_counts.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        threads,
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    const int clusters_to_load = C / n_groups;
    const int shared_mem = (E+1) * clusters_to_load * sizeof(float);

    clustered_broadcast_kernel<<<blocks, threads, shared_mem>>>(
        Y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        G.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        F.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        X.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        indx_maps.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        n_groups
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "clustered_aggregate",
        &clustered_aggregate,
        "Aggregate the vectors of X based on the "
        "indices in G multiplied by F."
    );
    m.def(
        "clustered_broadcast",
        &clustered_broadcast,
        "Broadcast the vectors of Y based on the indices "
        "in G multiplied by F back to X."
    );
}
