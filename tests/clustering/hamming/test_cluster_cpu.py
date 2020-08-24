#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch

from fast_transformers.clustering.hamming import cluster_cpu
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster


def generate_hash(n_points, d, b):
    x = torch.rand(n_points, d)
    a = torch.randn(b, d+1)
    a[:,d] = 0
    h = torch.zeros(n_points, dtype=torch.int64)
    compute_hashes(x, a, h)
    return h


class TestClusterCPU(unittest.TestCase):
    def test_long_clusters(self):
        for bits in range(1, 63):
            hashes = torch.cat([
                torch.zeros(50).long(),
                torch.ones(50).long() * (2**bits - 1)
            ]).view(1, 1, 100)[:,:,torch.randperm(100)]
            lengths = torch.full((1,), 100, dtype=torch.int32)
            centroids = torch.empty(1, 1, 2, dtype=torch.int64)
            clusters = torch.empty(1, 1, 100, dtype=torch.int32)
            counts = torch.empty(1, 1, 2, dtype=torch.int32)

            cluster_cpu(
                hashes,
                lengths,
                centroids,
                clusters,
                counts,
                10,
                bits
            )
            self.assertEqual(
                tuple(sorted(centroids.numpy().ravel().tolist())),
                (0, 2**bits - 1)
            )
            self.assertTrue(torch.all(counts==50))

    def test_two_clusters(self):
        hashes = torch.cat([
            torch.zeros(50).long(),
            torch.full((50,), 255, dtype=torch.int64)
        ]).view(1, 1, 100)[:,:,torch.randperm(100)]
        lengths = torch.full((1,), 100, dtype=torch.int32)
        centroids = torch.empty(1, 1, 2, dtype=torch.int64)
        clusters = torch.empty(1, 1, 100, dtype=torch.int32)
        counts = torch.empty(1, 1, 2, dtype=torch.int32)

        cluster_cpu(
            hashes,
            lengths,
            centroids,
            clusters,
            counts,
            10,
            8
        )
        self.assertEqual(
            tuple(sorted(centroids.numpy().ravel().tolist())),
            (0, 255)
        )
        self.assertTrue(torch.all(counts==50))

    def test_power_of_2_clusters(self):
        hashes = torch.cat([
            torch.full((10,), 1<<i, dtype=torch.int64)
            for i in range(8)
        ]).view(1, 1, 80)[:,:,torch.randperm(80)]
        lengths = torch.full((1,), 80, dtype=torch.int32)
        centroids = torch.empty(1, 1, 8, dtype=torch.int64)
        clusters = torch.empty(1, 1, 80, dtype=torch.int32)
        counts = torch.empty(1, 1, 8, dtype=torch.int32)

        cluster_cpu(
            hashes,
            lengths,
            centroids,
            clusters,
            counts,
            2000,
            8
        )
        self.assertEqual(
            tuple(sorted(centroids.numpy().ravel().tolist())),
            (1, 2, 4, 8, 16, 32, 64, 128)
        )
        self.assertTrue(torch.all(counts==10))

    def test_many_sequences(self):
        hashes = torch.cat([
            torch.zeros(50).long(),
            torch.full((50,), 255, dtype=torch.int64)
        ]).view(1, 1, 100)[:,:,torch.randperm(100)].repeat(5, 3, 1)
        lengths = torch.full((5,), 100, dtype=torch.int32)
        centroids = torch.empty(5, 3, 2, dtype=torch.int64)
        clusters = torch.empty(5, 3, 100, dtype=torch.int32)
        counts = torch.empty(5, 3, 2, dtype=torch.int32)

        cluster_cpu(
            hashes,
            lengths,
            centroids,
            clusters,
            counts,
            10,
            8
        )
        self.assertTrue(torch.all(centroids.min(2)[0] == 0))
        self.assertTrue(torch.all(centroids.max(2)[0] == 255))
        self.assertTrue(torch.all(counts==50))

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
            hashes = generate_hash(n_points, E, n_buckets).view(N, H, L)
            groups = torch.zeros((N, H, L), dtype=torch.int32)
            counts = torch.zeros((N, H, k), dtype=torch.int32)
            centroids = torch.zeros((N, H, k), dtype=torch.int64)
            distances = torch.zeros((N, H, L), dtype=torch.int32)
            cluster_bit_counts = torch.zeros((N, H, k, n_buckets),
                                             dtype=torch.int32)
            sequence_lengths = torch.ones((N,), dtype=torch.int32) * L
            sequence_lengths.random_(1, L+1)
                
            s = time.time()
            for i in range(50):
                cluster(
                    hashes, sequence_lengths,
                    groups=groups, counts=counts, centroids=centroids,
                    distances=distances, bitcounts=cluster_bit_counts,
                    iterations=n_iterations,
                    bits=n_buckets
                )
            e = time.time()
            t_clustering = e - s

            print("Clustering with {} bits took {} time".format(n_buckets,
                                                                t_clustering))


if __name__ == "__main__":
    unittest.main()
