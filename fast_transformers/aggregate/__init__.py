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
    from .clustered_broadcast_cuda import \
        clustered_broadcast as clustered_broadcast_gpu

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
    

def clustered_broadcast(Y, groups, counts, lengths, X=None):
    device = Y.device
    if X is None:
        X = torch.zeros(
            groups.shape + (Y.shape[-1],),
            device=device,
            dtype=Y.dtype
        )

    if device.type == "cpu":
        raise NotImplementedError
    else:
        N, H, C, E = Y.shape
        _, _, L, E = X.shape
   
        queries_per_block = min(L, 1024) 
        threads = queries_per_block
        blocks = (L//threads) + C + 1
        query_map = torch.ones((N, H, blocks),
                               dtype=torch.int32,
                               device=Y.device) * L 
        blocks_map = torch.ones_like(query_map,
                                     dtype=torch.int32,
                                     device=Y.device) * -1 
        _, sorted_group_indices = torch.sort(groups, descending=True, dim=-1)
        factors = torch.ones_like(counts, dtype=Y.dtype)
        clustered_broadcast_gpu(
            Y,
            groups,
            factors,
            X,
            lengths,
            blocks_map,
            query_map,
            counts,
            sorted_group_indices,
        )

    return X
