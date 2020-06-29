#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import numpy as np
import torch
import time

from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster

def simple_lsh(X, A):
    B = (torch.einsum("nj,ij->ni", [X, A]) > 0).long()
    bits = 2**torch.arange(A.shape[0])
    return torch.einsum("ni,i->n", [B, bits])


def generate_hash(n_points, d, b, h):
    torch.manual_seed(0)
    x = torch.rand(n_points, d).cuda()
    a = torch.randn(b, d + 1).cuda()
    compute_hashes(x, a, h)
    return h


def time_clustering(L, N, H, E,
                    n_batches, n_attentions,
                    k, n_buckets, n_iterations, verbose):
    n_points = L * N * H 
    hashes = torch.zeros(n_points, dtype=torch.int64).cuda()
    hashes = generate_hash(n_points, E, n_buckets, hashes).view(N, H, L)

    groups = torch.zeros((N, H, L), dtype=torch.int32).cuda()
    counts = torch.zeros((N, H, k), dtype=torch.int32).cuda()
    centroids = torch.zeros((N, H, k), dtype=torch.int64).cuda()
    distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()
    cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                     dtype=torch.int32).cuda()
    sequence_lengths = torch.ones((N,), dtype=torch.int32).cuda() * L
     
    start = time.time()
    for batch_idx in range(int(n_batches)):
        for attention_idx in range(int(n_attentions)):
            #hashes = generate_hash(n_points, E, n_buckets, hashes).view(L, N, H)
            cluster(
                hashes, sequence_lengths,
                groups=groups, counts=counts, centroids=centroids,
                distances=distances, bitcounts=cluster_bit_counts,
                iterations=n_iterations,
                bits=n_buckets
            )
    end = time.time()
    duration = end - start
    print("Time Elapsed: {}".format(duration))


if __name__ == "__main__":
    L = 1000
    N = 12
    H = 8
    E = 32

    n_batches = 50000/N
    n_attentions = 3
    
    k = 30
    n_buckets = 31
    n_iterations = 10
    verbose = 0
    
    time_clustering(L, N, H, E,
                    n_batches, n_attentions,
                    k, n_buckets, n_iterations, verbose)
