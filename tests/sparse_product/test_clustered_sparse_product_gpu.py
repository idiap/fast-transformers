#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
import time
import unittest

import torch
from torch.nn.init import normal_

from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.sparse_product import sparse_dot_product
from fast_transformers.sparse_product import clustered_sparse_dot_product

def cluster_queries(Q, query_lengths, C, I, B):
    N, H, L, E = Q.shape
    planes = Q.new_empty((B, E+1))
    normal_(planes)
    planes[:, -1] = 0
    hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)
    # Cluster the hashes and return the cluster index per query
    groups, counts =  cluster(
        hashes,
        query_lengths,
        clusters=C,
        iterations=I,
        bits=B
    )

    return groups, counts


class TestSparseProductCUDA(unittest.TestCase):
    @property
    def device(self):
        return "cuda"

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_simple_product(self):
        N = 2
        H = 2
        L = 1000
        E = 32
        S = 1000
        k = 32
        C = 50
        I = 5
        B = 16

        Q = torch.randn(N, H, L, E).to(self.device)
        K = torch.randn(N, H, S, E).to(self.device)
        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()
        products = clustered_sparse_dot_product(Q, K, topk, groups, counts, lengths)
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )

        all_products = torch.einsum("nhle,nhse->nhls", Q, K)
        products_2 = all_products[
            torch.arange(N).view(N, 1, 1, 1),
            torch.arange(H).view(1, H, 1, 1),
            torch.arange(L).view(1, 1, L, 1),
            topk_broadcast.long()
        ]

        self.assertLess(
            torch.max(torch.abs(
                products_2 - products
                )
            ),
            1e-4
        )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_small_benchmark(self):
        N = 12
        H = 8
        L = 1000
        E = 32
        S = 1000
        k = 32
        C = 100
        I = 10
        B = 32

        Q = torch.randn(N, H, L, E).to(self.device)
        K = torch.randn(N, H, S, E).to(self.device)
        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()
        products = torch.zeros((N, H, L, k), dtype=torch.float32).to(self.device)
        products = clustered_sparse_dot_product(Q, K, topk, groups, counts, lengths)
        for i in range(1000):
            products = clustered_sparse_dot_product(
                Q,
                K,
                topk,
                groups,
                counts,
                lengths
            )

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        products = clustered_sparse_dot_product(
            Q,
            K,
            topk,
            groups,
            counts,
            lengths
        )
        e.record()
        torch.cuda.synchronize()
        t_sc = s.elapsed_time(e)
   
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )

        for i in range(1000):
            products = sparse_dot_product(
                Q,
                K,
                topk_broadcast.long()
            )
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        products_s = sparse_dot_product(
            Q,
            K,
            topk_broadcast.long(),
        )
        e.record()
        torch.cuda.synchronize()
        t_s = s.elapsed_time(e)

        for i in range(1000):
            torch.einsum("nhle,nhse->nhls", Q, K)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.einsum("nhle,nhse->nhls", Q, K)
        e.record()
        torch.cuda.synchronize()
        t_f = s.elapsed_time(e)
        print("Sparse_Clustered: {}, Sparse: {}, Full: {}".format(t_sc, t_s, t_f))


if __name__ == "__main__":
    unittest.main()
