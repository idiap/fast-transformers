//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> int32_accessor_1d;
typedef torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> int32_accessor_3d;
typedef torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> int32_accessor_4d;
typedef torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> int64_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;


/**
 * Compute hamming distances
 */
__device__
int hamming_distance(int64_t a, int64_t b) {
    return  __popcll(a ^ b);
}


/**
 * Set up the kernel to generate cuda random numbers
 */
__global__
void setup_kernel(curandState *state) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}


/**
 * This kernel assigns datapoints to the closest centroids based on the hamming
 * distance
 *
 * Arguments:
 * ---------
 * Inputs:
 *     hash_codes             : hash codes tensor to be clustered
 *     lengths                : sequence lengths array
 *     centroids              : current estimate of the centroids
 *     n_blocks_per_sequence  : number of blocks allocated per sequence
 *     MAX                    : MAX distance possible (64 int_64 hamming)
 *
 * Outputs:
 *     labels                 : labels  to be assigned to each data point
 *     distances              : distances to the closest cluster
 */
__global__
void assign_clusters_kernel(
    const int64_accessor_3d hash_codes,
    const int32_accessor_1d lengths,
    const int64_accessor_3d centroids,
    int32_accessor_3d labels,
    int32_accessor_3d distances,
    const int n_blocks_per_sequence,
    int MAX=65
) {
    int H = centroids.size(1);
    int L = hash_codes.size(2);
    int K = centroids.size(2);

    // Load the shared memory
    const int sequence_index = blockIdx.x / n_blocks_per_sequence;
    const int n = sequence_index / H;
    const int h = sequence_index % H;

    extern __shared__ int64_t shared_means[];
    if (threadIdx.x < K) {
      shared_means[threadIdx.x] = centroids[n][h][threadIdx.x];
    }

    __syncthreads();

    // Extract the indexes
    const int l = ((blockIdx.x % n_blocks_per_sequence)*blockDim.x) + threadIdx.x;

    // Each block is only responsible for one sequence
    if(l >= L) {
        return;
    }

    // Beyond the sequence length set the cluster label to (K+1) where K is the clusters
    if(l >= lengths[n]) {
        labels[n][h][l] = K+1;
        distances[n][h][l] = -1;
        return;
    }

    // Make global loads once.
    const int64_t x = hash_codes[n][h][l];

    // update the cluster assingments
    // 64 bit hashcodes can have maximum hamming distance as 64
    int best_distance = MAX;
    int best_cluster = 0;
    int dist = 0;
    for (int cluster = 0; cluster < K; ++cluster) {
        dist = hamming_distance(x, shared_means[cluster]);
        if (dist < best_distance) {
            best_distance = dist;
            best_cluster = cluster;
        }
    }

    labels[n][h][l] = best_cluster;
    distances[n][h][l] = best_distance;
}

/**
 * This kernel counts the number of data points belonging to each  cluster and
 * also updates cluster_bit_counts for each cluster cluster_bit_counts for any
 * cluster is an array with size [B x 1]. Each position stores the
 * difference of number of data points with ones and number of data points with
 * zeros at that position in the binary representation of the number.
 *
 * Arguments:
 * ---------
 * Inputs:
 *     labels             : labels  to be assigned to each data point
 *     hash_codes         : hash codes to be clustered
 *
 * Outputs:
 *     counts             : array to store the number of datapoints
 *                          belonging to any cluster
 *     cluster_bit_counts : array containing the bit counts
 */
__global__
void bit_count_kernel(
    const int32_accessor_3d labels,
    const int64_accessor_3d hash_codes,
    int32_accessor_3d counts,
    int32_accessor_4d cluster_bit_counts
) {
    const int N = labels.size(0);
    const int H = labels.size(1);
    const int L = labels.size(2);
    const int K = counts.size(2);
    const int B = cluster_bit_counts.size(3);

    const int hl = H*L;
    // Extract the indices
    int full_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int sequence_index = full_idx / L;
    const int n = sequence_index / H;
    const int h = sequence_index % H;
    const int l = full_idx - n*hl - h*L;
    if (n >= N)
        return;

    const int64_t x = hash_codes[n][h][l];
    int val_to_add = -1;
    const int best_cluster = labels[n][h][l];
    if(best_cluster == (K+1)) {
        return;
    }

    for (int i=0; i<B; i++) {
        int64_t bit= 1LL << i;
        if((x & bit) > 0) {
            val_to_add = 1;
        }
        else {
            val_to_add = -1;
        }
        atomicAdd(&cluster_bit_counts[n][h][best_cluster][i], val_to_add);
    }
    atomicAdd(&counts[n][h][best_cluster], 1);
}


/**
 * This kernel computes the new means based on the cluster_bit_counts
 * Arguments:
 * ---------
 * Inputs:
 *     state              : cuda randome state for the random number generation
 *     counts             : array to store the number of datapoints
 *                          belonging to any cluster
 * Outputs:
 *     centroids          : centroids to be updated
 *     cluster_bit_counts : array containing the bit counts
 */
__global__
void compute_means_kernel(
    const int32_accessor_3d counts,
    int32_accessor_4d cluster_bit_counts,
    int64_accessor_3d centroids,
    curandState* state
) {
    const int N = counts.size(0);
    const int H = counts.size(1);
    const int K = counts.size(2);
    const int B = cluster_bit_counts.size(3);

    // Extract indices
    const int full_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( full_idx >= (K*N*H))
        return;

    const int sequence_idx = full_idx / K;
    const int n = sequence_idx / H;
    const int h = sequence_idx % H;
    const int k = full_idx % K;

    int64_t mean_k = 0;
    const int64_t MAX = (1LL << (B));

    // if the counts for the current cluster is 0 set mean to random
    if(counts[n][h][k] == 0) {
        centroids[n][h][k] = int64_t(curand(state + k) % MAX);
        return;
    }

    //update otherwise
    for( int i=0; i<B; i++) {
        if(cluster_bit_counts[n][h][k][i] == 0) {
            cluster_bit_counts[n][h][k][i] =
                (curand(state + k) & 1LL);
        }
        if(cluster_bit_counts[n][h][k][i] > 0) {
            mean_k = mean_k | (1LL << i);
        }
    }
    centroids[n][h][k] = mean_k;
}

/**
 * Kmeans runs lloyd iteratively to first assign the points and then update
 * the clusters
 * Arguments:
 * ---------
 * Inputs:
 *     hash_codes           : the hash codes to be clustered
 *     lengths            : sequence lengths array
 *     centroids              : centroid buffer
 *     distances          : distances buffer
 *     cluster_bit_counts : bit counts buffer
 *     iterations           : number of iterations of Lloyd
 *
 * Outputs:
 *     labels             : array to store the labels assigned to each point
 *     counts             : array to store the number of datapoints belonging
 *                            to any cluster
 *                            Size (L*NH*K)
 *                            [0..K-1] are counts for 1st sequence
 *                            [K..2K-1] are counts for 2nd sequence.
 */
void kmeans(
    const torch::Tensor hash_codes,
    const torch::Tensor lengths,
    torch::Tensor centroids,
    torch::Tensor distances,
    torch::Tensor cluster_bit_counts,
    torch::Tensor labels,
    torch::Tensor counts,
    int iterations
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(hash_codes.device());

    const int64_accessor_3d hash_codes_acc = hash_codes.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>();
    const int32_accessor_1d lengths_acc = lengths.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
    int64_accessor_3d centroids_acc = centroids.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>();
    int32_accessor_3d distances_acc = distances.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>();
    int32_accessor_4d cluster_bit_counts_acc = cluster_bit_counts.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
    int32_accessor_3d labels_acc = labels.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>();
    int32_accessor_3d counts_acc = counts.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>();

    const int N = hash_codes.size(0);
    const int H = hash_codes.size(1);
    const int NH = N*H;
    const int L = hash_codes.size(2);
    const int K = centroids.size(2);
    const int B = cluster_bit_counts.size(3);

    // allocate the temporary arrays we will need
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    setup_kernel<<<1,K>>>(d_state);

    // Estimate the number of threads we will need
    const int n_blocks_per_sequence = (L-1)/1024 + 1;
    // Dividing the number of threads roughly equally among blocks
    // Max because each blocks needs K threads to load shared memory
    const int n_threads_assign = max((L-1)/n_blocks_per_sequence + 1, K);
    const int n_blocks_assign = NH * n_blocks_per_sequence;
    const int shared_mem_assign = K * sizeof(int64_t);

    const int n_threads_cnt = 1024;
    const int n_blocks_cnt = ((L*NH)-1)/n_threads_cnt + 1;

    const int n_threads_centroids = 1024;
    const int n_blocks_centroids = ((K*NH)-1)/n_threads_cnt + 1;

    //Lloyd iterations
    for (size_t itr = 0; itr < iterations; ++itr) {
        assign_clusters_kernel<<<n_blocks_assign,
                                 n_threads_assign,
                                 shared_mem_assign>>>(
            hash_codes_acc,
            lengths_acc,
            centroids_acc,
            labels_acc,
            distances_acc,
            n_blocks_per_sequence
        );

        counts.zero_();
        cluster_bit_counts.zero_();

        bit_count_kernel<<<n_blocks_cnt,
                           n_threads_cnt>>>(
            labels_acc,
            hash_codes_acc,
            counts_acc,
            cluster_bit_counts_acc
        );
        compute_means_kernel<<<n_blocks_centroids,
                               n_threads_centroids>>>(
            counts_acc,
            cluster_bit_counts_acc,
            centroids_acc,
            d_state
        );
    }

    assign_clusters_kernel<<<n_blocks_assign,
                             n_threads_assign,
                             shared_mem_assign>>>(
        hash_codes_acc,
        lengths_acc,
        centroids_acc,
        labels_acc,
        distances_acc,
        n_blocks_per_sequence
    );

    counts.zero_();
    cluster_bit_counts.zero_();

    bit_count_kernel<<<n_blocks_cnt,
                       n_threads_cnt>>>(
        labels_acc,
        hash_codes_acc,
        counts_acc,
        cluster_bit_counts_acc
    );

    cudaFree(d_state);
    return;
}


/**
 * Cluster the hash codes H using Llyod's K-Means clustering
 * Inputs:
 *
 * Arguments:
 * ---------
 * Inputs:
 *     hashes    : hashes to be clustered
 *
 * Buffers:
 *     centroids : centroids buffer
 *     distances : distances buffer
 *     bitcounts : cluster_bit_counts buffer
 *
 * Outputs:
 *     clusters  : Store the groups/labels/assignments
 *     counts    : Store the counts of the number of points in each cluster
 */
void cluster(
    const torch::Tensor hashes,
    const torch::Tensor lengths,
    torch::Tensor centroids,
    torch::Tensor distances,
    torch::Tensor bitcounts,
    torch::Tensor clusters,
    torch::Tensor counts,
    int n_iterations,
    int B
) {
    int K = centroids.size(2);
    int N = hashes.size(0);
    int H = hashes.size(1);
    int L = hashes.size(2);

    // initialize the centroids
    //centroids.view({-1, K}) = hashes.view({-1, L}).narrow(1, 0, K);

    kmeans(
        hashes,
        lengths,
        centroids,
        distances,
        bitcounts,
        clusters,
        counts,
        n_iterations
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &cluster, "Cluster the hashed vectors by "
                               "performing a few iterations of k-means");
}
