//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> int_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_accessor_3d;
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;


__global__ void aggregate_kernel(
    const float_accessor_4d x,
    const int_accessor_3d g,
    const float_accessor_3d f,
    float_accessor_4d y
) {
    int N = x.size(0);
    int H = x.size(1);
    int L = x.size(2);
    int E = x.size(3);
    int C = y.size(2);

    // Extract all the indices
    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hl = H*L;
    int n = full_idx / hl;
    int h = (full_idx % hl) / L;
    int l = full_idx - n*hl - h*L;

    // Bail if out of bounds
    if (n >= N) {
        return;
    }

    // Copy-aggregate from x into y
    int k = g[n][h][l];
    if ((k < 0) || (k >= C)) {
        return;
    }
    float factor = f[n][h][k];
    for (int e=0; e<E; e++) {
        atomicAdd(&y[n][h][k][e], x[n][h][l][e] * factor);
    }
}


/**
 * Aggregate the passed vectors X based on group indices in G multiplied by
 * factors F.
 */
void aggregate(
    const torch::Tensor X,
    const torch::Tensor G,
    const torch::Tensor F,
    torch::Tensor Y
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(X.device());

    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);
    int C = Y.size(2);

    const int threads = 1024;
    int blocks = (L*N*H + threads - 1) / threads;

    aggregate_kernel<<<blocks, threads>>>(
        X.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        G.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        F.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        Y.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


__global__ void broadcast_kernel(
    const float_accessor_4d y,
    const int_accessor_3d g,
    const float_accessor_3d f,
    float_accessor_4d x
) {
    int N = x.size(0);
    int H = x.size(1);
    int L = x.size(2);
    int E = x.size(3);
    int C = y.size(2);

    // Extract all the indices
    int full_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hl = H*L;
    int n = full_idx / hl;
    int h = (full_idx % hl) / L;
    int l = full_idx - n*hl - h*L;

    // Bail if out of bounds
    if (n >= N) {
        return;
    }

    // Copy-broadcast from y into x
    int k = g[n][h][l];
    if ((k < 0) || (k >= C)) {
        return;
    }

    float factor = f[n][h][k];
    for (int e=0; e<E; e++) {
        x[n][h][l][e] = (y[n][h][k][e] * factor);
    }
}


/**
 * Broadcast the aggregated results from tensor Y back to tensor X based on
 * group indices G multiplied by factors F.
 */
void broadcast(
    const torch::Tensor Y,
    const torch::Tensor G,
    const torch::Tensor F,
    torch::Tensor X
) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(Y.device());

    int N = X.size(0);
    int H = X.size(1);
    int L = X.size(2);
    int E = X.size(3);
    int C = Y.size(2);

    const int threads = 1024;
    int blocks = (L*N*H + threads - 1) / threads;

    broadcast_kernel<<<blocks, threads>>>(
        Y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        G.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        F.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        X.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "aggregate",
        &aggregate,
        "Aggregate the vectors of X based on the "
        "indices in G multiplied by F."
    );
    m.def(
        "broadcast",
        &broadcast,
        "Broadcast the vectors of Y based on the indices "
        "in G multiplied by F back to X."
    );
}
