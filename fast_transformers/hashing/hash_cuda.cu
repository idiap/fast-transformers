//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <cassert>

#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>

typedef torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> int64_accessor;
typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float_accessor;


/**
 * Kernel to project the input queries "x" on to the hyperplanes "a";
 * compute hashes and write them to "h".
 * We use shared memory to load all the hyperplanes "a" once.
 * We avoid atomic OR operations by storing the proections
 * temporarily on shared memory and then doing a synchronized write to "h".
 * The kernel computes "queries_per_block" hashes in every block due
 * to synchronized writing.
 */
__global__ void hash_kernel(
    const float_accessor x,
    const float_accessor a,
    int64_accessor h,
    int queries_per_block
) {
    int N = x.size(0);
    int B = a.size(0);
    int D = x.size(1);

    // Since the planes are going to used for each data point
    // using shared memory to load all the planes once is very useful.
    extern __shared__ float shared_planes[];
    if (threadIdx.x < B) {
        int plane_idx = threadIdx.x;
        float *s_plane = shared_planes + threadIdx.x;
        for (int i=0; i<=D; i++) {
            *s_plane = a[plane_idx][i];
            s_plane += B;
        }
    }

    // Shared memory to store for projection for a plane
    // We do this to avoid atomic or for the hash.
    uint8_t *shared_out = (uint8_t*)(shared_planes + (B*(D+1)));
    __syncthreads();


    // Extract the indices
    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_idx / B;
    int b = full_idx % B;

    // Bail if the index is out of bounds
    if (n >= N) {
        return;
    }

    // Compute the dot product and the contributing bit
    float s = 0;
    float *plane = shared_planes + b;
    for (int i=0; i<D; i++) {
        s += x[n][i] * (*plane);
        plane = plane + B;
    }

    // Storing the dot product on the shared memory;
    shared_out[threadIdx.x] = uint8_t(s > *plane);
    __syncthreads();

    // Aggregating the result and writing on to the hash
    if (threadIdx.x < queries_per_block) {
        int n = blockIdx.x * queries_per_block + threadIdx.x;
        if (n >= N) {
            return;
        }
        unsigned long long int h_out = 0;
        for (int b=0; b<B; b++) {
            unsigned long long int bit = static_cast<unsigned long long int>(
                shared_out[threadIdx.x*B + b]) << b;
            h_out = (h_out | bit);
        }
        h[n] = h_out;
    }
}


/**
 * Hash the vectors in X with the hyperplanes A and store the result in H.
 * The positive side of the plane gets a 1 the negative a 0.
 */
void compute_hashes(torch::Tensor X, torch::Tensor A, torch::Tensor H) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(X.device());

    int N = X.size(0);
    int B = A.size(0);
    int D = X.size(1);
    assert(((void)"Bias expected for the parameters", D+1 == A.size(1)));

    // Computing the number of hashes to be computed per block
    const int max_threads = 1024;
    const int max_queries_per_block = 512;
    const int threads = (max_queries_per_block * B) < max_threads ?
        (max_queries_per_block * B): (max_threads / B) * B;
    int queries_per_block = threads / B;

    // Allocating the shared memory to store the hyperplanes and output
    const int shared_mem_planes = (B * (D+1) * sizeof(float)) +
                                  (queries_per_block * B * sizeof(uint8_t));
    int blocks = (N*B + threads - 1) / threads;

    hash_kernel<<<blocks, threads, shared_mem_planes>>>(
        X.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        H.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        queries_per_block
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_hashes",
        &compute_hashes,
        "Hash the vectors X using SIMPLE-LSH."
    );
}
