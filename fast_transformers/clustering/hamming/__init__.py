#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import numpy as np

import torch

from .cluster_cpu import cluster as cluster_cpu
try:
    from .cluster_cuda import cluster as cluster_gpu
except ImportError:
    pass


def cluster(
    hashes,
    lengths,
    groups=None,
    counts=None,
    centroids=None,
    distances=None,
    bitcounts=None,
    clusters=30,
    iterations=10,
    bits=32
):
    """Cluster hashes using a few iterations of K-Means with hamming distance.

    All the tensors default initialized to None are optional buffers to avoid
    memory allocations. distances and bitcounts are only used by the CUDA
    version of this call. clusters will be ignored if centroids is provided.

    Arguments
    ---------
        hashes: A long tensor of shape (N, H, L) containing a hashcode for each
                query.
        lengths: An int tensor of shape (N,) containing the sequence length for
                 each sequence in hashes.
        groups: An int tensor buffer of shape (N, H, L) contaning the cluster
                in which the corresponding hash belongs to.
        counts: An int tensor buffer of shape (N, H, K) containing the number
                of elements in each cluster.
        centroids: A long tensor buffer of shape (N, H, K) containing the
                   centroid for each cluster.
        distances: An int tensor of shape (N, H, L) containing the distance to
                   the closest centroid for each hash.
        bitcounts: An int tensor of shape (N, H, K, bits) containing the number
                   of elements that have 1 for a given bit.
        clusters: The number of clusters to use for each sequence. It is
                  ignored if centroids is not None.
        iterations: How many k-means iterations to perform.
        bits: How many of the least-significant bits in hashes to consider.

    Returns
    -------
        groups and counts as defined above.
    """
    device = hashes.device
    N, H, L = hashes.shape

    # Unfortunately cpu and gpu have different APIs so the entire call must be
    # surrounded by an if-then-else
    if device.type == "cpu":
        if groups is None:
            groups = torch.empty((N, H, L), dtype=torch.int32)
        if centroids is None:
            centroids = torch.empty((N, H, clusters), dtype=torch.int64)
            centroids = hashes[:, :, np.random.choice(L, size=[clusters], replace=False)]
        K = centroids.shape[2]
        if counts is None:
            counts = torch.empty((N, H, K), dtype=torch.int32)

        cluster_cpu(
            hashes, lengths,
            centroids, groups, counts,
            iterations, bits
        )

        return groups, counts

    else:
        if groups is None:
            groups = torch.empty((N, H, L), dtype=torch.int32, device=device)
        if centroids is None:
            centroids = torch.empty((N, H, clusters), dtype=torch.int64,
                                    device=device)
            centroids = hashes[:, :, np.random.choice(L, size=[clusters], replace=False)]
        K = centroids.numel() // N // H
        #K = clusters
        if counts is None:
            counts = torch.empty((N, H, K), dtype=torch.int32, device=device)
        if distances is None:
            distances = torch.empty((N, H, L), dtype=torch.int32,
                                    device=device)
        if bitcounts is None:
            bitcounts = torch.empty((N, H, K, bits), dtype=torch.int32,
                                    device=device)
        groups = groups.view(N, H, L)
        counts = counts.view(N, H, K)
        centroids = centroids.view(N, H, K)
        distances = distances.view(N, H, L)
        bitcounts = bitcounts.view(N, H, K, -1)

        cluster_gpu(
            hashes, lengths,
            centroids, distances, bitcounts, groups, counts,
            iterations, bits
        )

        return groups, counts
        
