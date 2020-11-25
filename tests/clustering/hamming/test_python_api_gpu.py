#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import numpy as np

import torch

from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster

def simple_lsh(X, A):
    B = (torch.einsum("nj,ij->ni", [X, A]) > 0).long()
    bits = 2**torch.arange(A.shape[0])
    return torch.einsum("ni,i->n", [B, bits])


def generate_hash(n_points, d, b):
    x = torch.rand(n_points, d)
    a = torch.randn(b, d+1)
    a[:,d] = 0
    h = torch.zeros(n_points, dtype=torch.int64)
    compute_hashes(x, a, h)
    return h


def hamming_distance(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def get_hamming_distances(data, closest_clusters, n_buckets):
    binary_repr_vectorized = np.vectorize(np.binary_repr)
    hamming_distance_vectorized = np.vectorize(hamming_distance)
    b_data = binary_repr_vectorized(data, n_buckets+1)
    b_closest_clusters = binary_repr_vectorized(closest_clusters, n_buckets+1)
    distance = hamming_distance_vectorized(b_data, b_closest_clusters)
    return distance


def labels_to_centroids(labels, centroids):
    labels[labels == (centroids.shape[-1] + 1)] = -1
    res = [list(map(c.__getitem__, l)) for c,l in zip(centroids,labels)]
    return np.asarray(res)


def verify_distances(data, labels, centroids, distances, n_buckets, lengths):
    closest_centroids = labels_to_centroids(labels, centroids)
    distances_np = get_hamming_distances(data, closest_centroids, n_buckets)
    for idx, l in enumerate(lengths):
        if( l < distances_np.shape[1] ):
            distances_np[idx,l:] = -1

    assert(np.all(distances_np==distances))


class TestClusteringGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_clustering_convergence(self):
        N=50
        H=4 
        E=32 
        n_iterations=10
    
        for n_buckets in range(1, 10):
            if os.getenv("VERBOSE_TESTS", ""):
                print('Testing convergence for {} bits'.format(n_buckets))
            k = 2**n_buckets

            L=k
            n_points = L * N * H
            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L).cuda()
            lengths = torch.ones((N,), dtype=torch.int32).cuda() * L
            distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()

            cluster(
                hashes, lengths,
                distances=distances,
                clusters=k,
                iterations=n_iterations,
                bits=n_buckets
            )
    
            distances_np = distances.data.cpu().numpy()
            self.assertEqual(distances_np.sum(), 0)

    def test_clustering(self):
        N=50
        H=4
        L=100
        E=32 

        k=20
        n_buckets=31
        n_iterations=10

        n_points = L * N * H

        groups = torch.zeros((N, H, L), dtype=torch.int32).cuda()
        counts = torch.zeros((N, H, k), dtype=torch.int32).cuda()
        centroids = torch.zeros((N, H, k), dtype=torch.int64).cuda()
        distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()
        cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                         dtype=torch.int32).cuda()
        sequence_lengths = torch.ones((N,), dtype=torch.int32).cuda() * L

        for i in range(50):
            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L).cuda()
    
            cluster(
                hashes, sequence_lengths,
                groups=groups, counts=counts, centroids=centroids,
                distances=distances, bitcounts=cluster_bit_counts,
                iterations=n_iterations,
                bits=n_buckets
            )
    
            lengths_np = sequence_lengths.repeat_interleave(H).cpu().numpy()
            hashes_np = hashes.view(N * H, L).cpu().numpy()
            groups_np = groups.view(N * H, L).cpu().numpy()
            distances_np = distances.view(N * H, L).cpu().numpy()
            centroids_np = centroids.view(N * H, k).cpu().numpy()
    
            verify_distances(hashes_np, groups_np, centroids_np,
                             distances_np, n_buckets, lengths_np)

    def test_masked_clustering(self):
        N=50
        H=4 
        L=100
        E=32 
        
        k=20
        n_buckets=31
        n_iterations=20

        n_points = L * N * H

        groups = torch.zeros((N, H, L), dtype=torch.int32).cuda()
        counts = torch.zeros((N, H, k), dtype=torch.int32).cuda()
        centroids = torch.zeros((N, H, k), dtype=torch.int64).cuda()
        distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()
        cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                         dtype=torch.int32).cuda()
        sequence_lengths = torch.ones((N,), dtype=torch.int32).cuda() * L
    
        for i in range(50):
            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L).cuda()
            sequence_lengths.random_(1, L+1)

            cluster(
                hashes, sequence_lengths,
                groups=groups, counts=counts, centroids=centroids,
                distances=distances, bitcounts=cluster_bit_counts,
                iterations=n_iterations,
                bits=n_buckets
            )
            lengths_np = sequence_lengths.repeat_interleave(H).data.cpu().numpy()
            hashes_np = hashes.view(N * H, L).data.cpu().numpy()
            groups_np = groups.view(N * H, L).data.cpu().numpy()
            distances_np = distances.view(N * H, L).data.cpu().numpy()
            centroids_np = centroids.view(N * H, k).data.cpu().numpy()
            verify_distances(hashes_np, groups_np, centroids_np, 
                             distances_np, n_buckets, lengths_np)

    def test_masked_clustering_convergence(self):
        N=50
        H=4 
        E=32 
        n_iterations=30
    
        for n_buckets in range(1, 10):
            if os.getenv("VERBOSE_TESTS", ""):
                print('Testing convergence for {} bits'.format(n_buckets))
            k = 2**n_buckets
            L = k + 1
            n_points = L * N * H

            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L).cuda()

            groups = torch.zeros((N, H, L), dtype=torch.int32).cuda()
            counts = torch.zeros((N, H, k), dtype=torch.int32).cuda()
            centroids = torch.zeros((N, H, k), dtype=torch.int64).cuda()
            distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()
            cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                             dtype=torch.int32).cuda()
            sequence_lengths = torch.ones((N,), dtype=torch.int32).cuda() * L
            sequence_lengths.random_(L)
            sequence_lengths += 1

            cluster(
                hashes, sequence_lengths,
                groups=groups, counts=counts, centroids=centroids,
                distances=distances, bitcounts=cluster_bit_counts,
                iterations=n_iterations,
                bits=n_buckets
            )

            lengths_np = sequence_lengths.cpu().numpy()
            distances_np = distances.cpu().numpy()
            for n in range(N):
                distances_np[n, :, lengths_np[n]-1:] = 0
            assert(distances_np.sum() == 0)

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_clustering(self):
        N=12
        H=4 
        L=1000
        E=32 
        
        k=100
        n_buckets=63
        n_iterations=10
        
        n_points = L * N * H
        for n_buckets in range(10, 64):        
            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L).cuda()
            groups = torch.zeros((N, H, L), dtype=torch.int32).cuda()
            counts = torch.zeros((N, H, k), dtype=torch.int32).cuda()
            centroids = torch.zeros((N, H, k), dtype=torch.int64).cuda()
            distances = torch.zeros((N, H, L), dtype=torch.int32).cuda()
            cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                             dtype=torch.int32).cuda()
            sequence_lengths = torch.ones((N,), dtype=torch.int32).cuda() * L
            sequence_lengths.random_(1, L+1)

            for i in range(500):
                cluster(
                    hashes, sequence_lengths,
                    groups=groups, counts=counts, centroids=centroids,
                    distances=distances, bitcounts=cluster_bit_counts,
                    iterations=n_iterations,
                    bits=n_buckets
                )

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            cluster(
                hashes, sequence_lengths,
                groups=groups, counts=counts, centroids=centroids,
                distances=distances, bitcounts=cluster_bit_counts,
                iterations=n_iterations,
                bits=n_buckets
            )   
            e.record()
            torch.cuda.synchronize()
            t_clustering = s.elapsed_time(e)

            print("Clustering with {} bits took {} time".format(
                n_buckets,
                t_clustering)
            )


if __name__ == "__main__":
    unittest.main()

