#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
from os import getenv
import unittest

import torch
from torch.nn.init import normal_

from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
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


class TestSparseProductBackward(unittest.TestCase):
    @property
    def device(self):
        return "cuda"

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def _zero_grad(self, Q, K):
        for x in [Q, K]:
            if x.grad is not None:
                x.grad[...] = 0

    def test_simple_grad(self):
        N = 2
        H = 2
        L = 1000
        E = 32
        S = 1000
        k = 32
        C = 50
        I = 5
        B = 16

        Q = torch.randn(N, H, L, E).to(self.device).requires_grad_(True)
        K = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)

        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )


        self._zero_grad(Q, K)
        QK_full = torch.einsum("nhle,nhse->nhls", Q, K)
        QK_selected = QK_full[
            torch.arange(N).view(N, 1, 1, 1).to(self.device),
            torch.arange(H).view(1, H, 1, 1).to(self.device),
            torch.arange(L).view(1, 1, L, 1).to(self.device),
            topk_broadcast.long()
        ]

        QK_selected.sum().backward()
        grad = [torch.clone(Q.grad), torch.clone(K.grad)]


        self._zero_grad(Q, K)
        QK_selected_hat = clustered_sparse_dot_product(
            Q, K, topk,
            groups, counts,
            lengths
        )

        QK_selected_hat.sum().backward()
        grad_hat = [torch.clone(Q.grad), torch.clone(K.grad)]

        self.assertLess(
            torch.abs(QK_selected - QK_selected_hat).max(),
            1e-4
        )
        for g1, g2 in zip(grad, grad_hat):
            self.assertLess(
                torch.abs(g1 - g2).max(),
                1e-4
            )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_backward(self):
        N = 12
        H = 8
        L = 1024
        S = 1024
        E = 32
        k = 32
        C = 100
        I = 10
        B = 32

        Q = torch.randn(N, H, L, E).to(self.device).requires_grad_(True)
        K = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()

        self._zero_grad(Q, K)
        for i in range(2000):
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            QK.sum().backward()
        self._zero_grad(Q, K)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        QK = torch.einsum("nhle,nhse->nhls", Q, K)
        s.record()
        QK.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_full = s.elapsed_time(e)

        self._zero_grad(Q, K)
        for i in range(2000):
            QK = clustered_sparse_dot_product(
                Q, K, topk,
                groups, counts,
                lengths
            )
            QK.sum().backward()
        self._zero_grad(Q, K)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        QK = clustered_sparse_dot_product(
            Q, K, topk,
            groups, counts,
            lengths
        )
        s.record()
        QK.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_sparse = s.elapsed_time(e)
        print("Benchmark Backward: T_Full: {}, T_Sparse: {}".format(t_full, t_sparse))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward_backward(self):
        N = 12
        H = 8
        L = 1024
        S = 1024
        E = 32
        k = 32
        C = 100
        I = 10
        B = 32

        Q = torch.randn(N, H, L, E).to(self.device).requires_grad_(True)
        K = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()

        self._zero_grad(Q, K)
        for i in range(2000):
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            QK.sum().backward()
        self._zero_grad(Q, K)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        QK = torch.einsum("nhle,nhse->nhls", Q, K)
        QK.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_full = s.elapsed_time(e)

        self._zero_grad(Q, K)
        for i in range(2000):
            QK = clustered_sparse_dot_product(
                Q, K, topk,
                groups, counts,
                lengths
            )
            QK.sum().backward()
        self._zero_grad(Q, K)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        QK = clustered_sparse_dot_product(
            Q, K, topk,
            groups, counts,
            lengths
        )
        QK.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_sparse = s.elapsed_time(e)
        print("Benchmark Forward-Backward: T_Full: {}, T_Sparse: {}".format(t_full, t_sparse))

if __name__ == "__main__":
    unittest.main()
