//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//


#include <array>
#include <random>

#include <torch/extension.h>


#ifdef __GNUC__
    inline int popcnt(int64_t x) {
        return __builtin_popcountll(x);
    }
#else
    #error "Popcnt not implemented"
#endif


/**
 * PyTorch <1.6.0 and >=1.6.0 are incompatible with respect to random number
 * generation. Thus we roll our own.
 */
struct ThreadSafePRNG {
    ThreadSafePRNG(uint64_t start, uint64_t end) :
        gen(std::random_device()()), dis(start, end) {}

    uint64_t random() {
        std::lock_guard<std::mutex> lock(mutex);

        return dis(gen);
    }

    private:
        std::mt19937 gen;
        std::uniform_int_distribution<uint64_t> dis;
        std::mutex mutex;
};


/**
 * Initialize with the first K hashes.
 */
void initialize(const torch::Tensor hashes, torch::Tensor centroids) {
    centroids.slice() = hashes.narrow(2, 0, centroids.size(2));
}


/**
 * For each hash compute the closest centroid and write it into the clusters
 * tensor.
 */
void assign(
    const torch::Tensor hashes,
    const torch::Tensor lengths,
    const torch::Tensor centroids,
    torch::Tensor clusters
) {
    int N = hashes.size(0);
    int H = hashes.size(1);
    int K = centroids.size(2);

    auto hash_a = hashes.accessor<int64_t, 3>();
    auto length_a = lengths.accessor<int32_t, 1>();
    auto centroid_a = centroids.accessor<int64_t, 3>();
    auto cluster_a = clusters.accessor<int32_t, 3>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        int maxl = length_a[n];
        for (int l=0; l<maxl; l++) {
            for (int h=0; h<H; h++) {
                int64_t hash = hash_a[n][h][l];
                int mind = 1000;
                int assignment = -1;
                for (int k=0; k<K; k++) {
                    int d = popcnt(hash ^ centroid_a[n][h][k]);
                    if (d < mind) {
                        mind = d;
                        assignment = k;
                    }
                }
                cluster_a[n][h][l] = assignment;
            }
        }
    }
}


/**
 * Recompute the centroids in a way such that the hamming distance from the
 * points in the cluster is minimized.
 */
void recompute_centroids(
    const torch::Tensor clusters,
    const torch::Tensor hashes,
    const torch::Tensor lengths,
    torch::Tensor centroids,
    int bits
) {
    int N = hashes.size(0);
    int H = hashes.size(1);
    int K = centroids.size(2);

    auto cluster_a = clusters.accessor<int32_t, 3>();
    auto hash_a = hashes.accessor<int64_t, 3>();
    auto length_a = lengths.accessor<int32_t, 1>();
    auto centroid_a = centroids.accessor<int64_t, 3>();

    ThreadSafePRNG prng(0, (1UL<<bits)-1UL);  // see the class comment on why

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        // The counts variable is keeping track of how many 1s and 0s we have in
        // each hash belonging to the same cluster. Then the centroid is the one
        // that has 1 where most hashes have 1 and 0 otherwise.
        // NOTE: Making it an array keeps it on the stack although vector<int>
        //       probably is exactly the same
        std::array<int, 64> counts;
        int maxl = length_a[n];
        for (int h=0; h<H; h++) {
            for (int k=0; k<K; k++) {
                // Zero the counts to compute anew for this cluster and set the
                // flag that this centroid has no hashes assigned to it
                bool empty_centroid = true;
                for (auto &v : counts) {
                    v = 0;
                }

                // For each cluster (outer loop) we check every hash (probably
                // non-optimal). We can allocate K times more memory and do it
                // in one loop.
                for (int l=0; l<maxl; l++) {
                    if (cluster_a[n][h][l] == k) {
                        int64_t hash = hash_a[n][h][l];
                        for (int b=0; b<bits; b++) {
                            if (hash & 1L<<b) {
                                counts[b]++;
                            } else {
                                counts[b]--;
                            }
                        }
                        empty_centroid = false;
                    }
                }

                // If the centroid is non empty set the value based on the
                // counts else set it to random. We use the pytorch PRNG so
                // that it can be seeded from python.
                if (!empty_centroid) {
                    int64_t c = 0;
                    for (int b=0; b<bits; b++) {
                        c |= (int64_t(counts[b] > 0)) << b;
                    }
                    centroid_a[n][h][k] = c;
                } else {
                    int64_t c = prng.random();
                    centroid_a[n][h][k] = c & ((1L<<bits)-1);
                }
            }
        }
    }
}


/**
 * Simply count how many hashes are assigned to each cluster.
 */
void count_cluster_population(
    const torch::Tensor clusters,
    const torch::Tensor lengths,
    torch::Tensor counts
) {
    int N = clusters.size(0);
    int H = clusters.size(1);

    auto cluster_a = clusters.accessor<int32_t, 3>();
    auto length_a = lengths.accessor<int32_t, 1>();
    auto count_a = counts.accessor<int32_t, 3>();

    #pragma omp parallel for
    for (int n=0; n<N; n++) {
        int maxl = length_a[n];
        for (int l=0; l<maxl; l++) {
            for (int h=0; h<H; h++) {
                int k = cluster_a[n][h][l];
                if (k >= 0) {
                    count_a[n][h][k]++;
                }
            }
        }
    }
}


/**
 * Cluster the hashed vectors H by performing a few iterations of k-means.
 *
 * Arguments
 * ---------
 *     hashes: Tensor of int64_t and shape (N, H, L) containing the hashes
 *     lengths: Tensor of int32_t and shape (N,) containing the length of valid
 *              hashes for each of N independent inputs
 *     centroids: Uninitialized tensor of int64_t and shape (N, H, K)
 *     clusters: Uninitialized tensor of int64_t and shape (N, H, L) containing
 *               the cluster assignments for each hash
 *     counts: Uninitialized tensor of int32_t and shape (N, H, K) containing
 *             the number of elements in each cluster.
 *     iterations: How many k-means iterations to run
 *     bits: How many of the least significant bits to consider from the hashes
 */
void cluster(
    const torch::Tensor hashes,
    const torch::Tensor lengths,
    torch::Tensor centroids,
    torch::Tensor clusters,
    torch::Tensor counts,
    int iterations,
    int bits
) {
    // Initialize the centroids and the assignments
    initialize(hashes, centroids);
    clusters.fill_(-1);

    // Perform iterations of Lloyd's algorithm
    for (int i=1; i<iterations; i++) {
        assign(hashes, lengths, centroids, clusters);
        recompute_centroids(clusters, hashes, lengths, centroids, bits);
    }
    assign(hashes, lengths, centroids, clusters);

    // Compute the counts
    counts.fill_(0);
    count_cluster_population(clusters, lengths, counts);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &cluster, "Cluster the hashed vectors by "
                               "performing a few iterations of k-means");
}
