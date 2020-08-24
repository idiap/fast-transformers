#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
from os import getenv
import time
import unittest

import torch
from torch.nn.init import normal_

from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.sparse_product import clustered_sparse_weighted_average


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


class TestSparseWeightedAverage(unittest.TestCase):
    @property
    def device(self):
        return "cpu"

    def _zero_grad(self, Q, K):
        for x in [Q, K]:
            if x.grad is not None:
                x.grad[...] = 0

    def test_correctness(self):
        N = 2
        H = 4
        L = 3000
        S = 3000
        E = 32
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
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )

        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        self._zero_grad(weights, values)
        values_selected = values[
            torch.arange(N).view(N, 1, 1, 1).to(self.device),
            torch.arange(H).view(1, H, 1, 1).to(self.device),
            topk_broadcast.long()
        ]
        output = (weights.unsqueeze(-1)*values_selected).sum(-2)
        output.sum().backward()
        grad = [torch.clone(weights.grad), torch.clone(values.grad)]

        self._zero_grad(weights, values)
        output_hat = clustered_sparse_weighted_average(weights, values, topk, groups)
        output_hat.sum().backward()
        grad_hat = [torch.clone(weights.grad), torch.clone(values.grad)]
        self.assertLess(
            torch.abs(output - output_hat).max(),
            1e-4
        )
        for g1, g2 in zip(grad, grad_hat):
            self.assertLess(
                torch.abs(g1 - g2).max(),
                1e-3
            )

    def test_forward(self):
        N = 5
        H = 2 
        L = 100
        S = 100
        E = 32
        C = 10
        I = 10
        B = 32
        k = 5

        Q = torch.randn(N, H, L, E).to(self.device)
        K = torch.randn(N, H, S, E).to(self.device)
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

        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        values_selected = values[
            torch.arange(N).view(N, 1, 1, 1).to(self.device),
            torch.arange(H).view(1, H, 1, 1).to(self.device),
            topk_broadcast.long()
        ]

        output = (weights.unsqueeze(-1)*values_selected).sum(-2)
        output_hat = clustered_sparse_weighted_average(weights, values, topk, groups)
        self.assertLess(
            torch.abs(output - output_hat).max(),
            1e-4
        )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_small_forward(self):
        N = 12
        H = 8
        L = 2000
        S = 2000
        E = 32
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
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )

        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
       
        n_runs = 20
        s = time.time()
        for i in range(n_runs):
            output_hat = clustered_sparse_weighted_average(
                weights, values,
                topk, groups
            )
        e = time.time()
        t_sparse = (e - s) / n_runs
        print('T_sparse Forward:{}'.format(t_sparse))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_small_forward_backward(self):
        N = 12
        H = 8
        L = 2000
        S = 2000
        E = 32
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
        topk_broadcast = broadcast(
            topk.float(),
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, k), device=Q.device)
        )
        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)

        self._zero_grad(weights, values)
        n_runs = 20
        s = time.time()
        for i in range(n_runs):
            output_hat = clustered_sparse_weighted_average(
                weights, values,
                topk, groups
            )
            output_hat.sum().backward()
        e = time.time()
        t_sparse = (e - s) / n_runs
        print('T_sparse Forward Backward:{}'.format(t_sparse))


if __name__ == "__main__":
    unittest.main()
