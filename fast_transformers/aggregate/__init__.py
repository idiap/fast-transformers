#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import torch

from .aggregate_cpu import aggregate as aggregate_cpu, \
    broadcast as broadcast_cpu
try:
    from .aggregate_cuda import aggregate as aggregate_gpu, \
        broadcast as broadcast_gpu
    from .clustered_aggregate_cuda import \
        clustered_broadcast as clustered_broadcast_gpu, \
        clustered_aggregate as clustered_aggregate_gpu

except ImportError:
    pass


def aggregate(X, G, F, Y=None):
    device = X.device
    if Y is None:
        Y = torch.zeros(
            F.shape + (X.shape[-1],),
            device=device,
            dtype=X.dtype
        )
    else:
        Y.zero_()

    if device.type == "cpu":
        aggregate_cpu(X, G, F, Y)
    else:
        aggregate_gpu(X, G, F, Y)

    return Y


def broadcast(Y, G, F, X=None):
    device = Y.device
    if X is None:
        X = torch.zeros(
            G.shape + (Y.shape[-1],),
            device=device,
            dtype=Y.dtype
        )

    if device.type == "cpu":
        broadcast_cpu(Y, G, F, X)
    else:
        broadcast_gpu(Y, G, F, X)

    return X


# Divide the cluster into groups of equal size
# as constrained by the shared memory
def set_group(C, E):
    C_per_block = int(192 * 64 / (E+1))
    G_min = (C + C_per_block - 1) // C_per_block
    for G in range(G_min, C+1):
        if C % G == 0:
            return G


def clustered_broadcast(Y, groups, counts, factors, X=None):
    device = Y.device
    if X is None:
        X = torch.zeros(
            groups.shape + (Y.shape[-1],),
            device=device,
            dtype=Y.dtype
        )
    if device.type == "cpu":
        broadcast_cpu(Y, groups, factors, X)
    else:
        N, H, C, E = Y.shape
        _, _, L, _ = X.shape

        # Following are some booking keeping parameters to facilitate the
        # broadcast kernel that takes advantage of clustering
        # More information can be found in the cuda file
        with torch.no_grad():
            threads = 256
            G = set_group(C, E)
            group_counts = counts.view(N, H, G, -1).sum(-1)
            block_counts = (group_counts + threads - 1) // threads
            total_blocks = block_counts.sum().item()
            indx_maps = torch.ones(
                (total_blocks, 5),
                device=X.device,
                dtype=torch.int32
            )

        clustered_broadcast_gpu(
            Y,
            groups,
            factors,
            X,
            block_counts.int(),
            group_counts.int(),
            threads,
            G,
            total_blocks,
            indx_maps
        )
    return X


def clustered_aggregate(X, G, F, lengths, Y=None):
    device = X.device
    if Y is None:
        Y = torch.zeros(
            F.shape + (X.shape[-1],),
            device=device,
            dtype=X.dtype
        )
    else:
        Y.zero_()

    if device.type == "cpu":
        aggregate_cpu(X, G, F, Y)
    else:
        clustered_aggregate_gpu(X, G, F, lengths, Y)
    return Y
