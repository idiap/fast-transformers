#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
from os import getenv
import unittest

import numpy as np

import torch
from torch.nn.init import normal_

from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.sparse_product import clustered_sparse_weighted_average
from fast_transformers.sparse_product import clustered_sparse_dot_product
from fast_transformers.masking import LengthMask


def cluster_queries(Q, query_lengths, C, I, B):
    N, H, L, E = Q.shape
    planes = Q.new_empty((B, E+1))
    normal_(planes)
    planes[:, -1] = 0
    hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)
    # Cluster the hashes and return the cluster index per query
    groups, counts = cluster(
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

    def test_correctness_masked(self):
        N = 12
        H = 6
        L = 1000
        S = 1000
        E = 32
        k = 32
        C = 100
        I = 10
        B = 32
        for exp in range(30):
            N = np.random.randint(1, 6)
            H = np.random.randint(1, 8)
            C = np.random.randint(10, 500)
            L = np.random.randint(C, 2000)
            E = np.random.randint(10, 128)
            S = np.random.randint(100, 1000)
            k = np.random.randint(10, 64)

            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing Masked: N H L S E C k: "
                       "{} {} {} {} {} {} {}").format(N, H, L, S, E, C, k))

            Q = torch.randn(N, H, L, E).to(self.device)
            K = torch.randn(N, H, S, E).to(self.device)

            lengths = np.random.randint(C, L+1, N)
            lengths = torch.tensor(lengths, dtype=torch.int32).to(self.device)
            lengths[0] = L
            query_lengths = LengthMask(
                lengths,
                max_len=L
            )
            groups, counts = cluster_queries(Q, lengths, C, I, B)

            sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

            q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)
            s_queries = Q.reshape(-1, E).index_select(
                0, q_flat).view(N, H, L, E)
            Q_grouped = aggregate(
                s_queries, sorted_g.view(N, H, L), 1/counts.float()
            )

            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
            _, topk = torch.topk(QK, k, dim=-1)
            topk = topk.contiguous()
            topk_broadcast = broadcast(
                topk.float(),
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, k), device=Q.device)
            )
            weights_sorted = torch.rand(
                N, H, L, k).to(self.device).requires_grad_(True)
            weights_sorted.retain_grad()

            q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
            weights = torch.clone(
                weights_sorted.reshape(-1, k).index_select(
                    0, q_rev_flat
                ).view(N, H, L, k)
            )
            weights.retain_grad()

            values = torch.randn(
                N, H, S, E).to(self.device).requires_grad_(True)
            self._zero_grad(weights, values)
            values_selected = values[
                torch.arange(N).view(N, 1, 1, 1).to(self.device),
                torch.arange(H).view(1, H, 1, 1).to(self.device),
                topk_broadcast.long()
            ]
            output = (weights.unsqueeze(-1)*values_selected).sum(-2)
            output = output * query_lengths.float_matrix[:, None, :, None]
            output.sum().backward()
            grad = [torch.clone(weights.grad), torch.clone(values.grad)]

            self._zero_grad(weights_sorted, values)
            self._zero_grad(weights, values)

            output_hat_sorted = clustered_sparse_weighted_average(
                weights_sorted, values, topk, sorted_g.view(N, H, L), counts
            )
            output_hat = output_hat_sorted.reshape(
                -1, E).index_select(0, q_rev_flat).view(N, H, L, E)

            self.assertLess(
                torch.abs(output - output_hat).max(),
                1e-4
            )
            output_hat.sum().backward()
            weights_grad_sorted = torch.clone(weights_sorted.grad)
            weights_grad = torch.clone(
                weights_grad_sorted.reshape(-1, k).index_select(
                    0, q_rev_flat).view(N, H, L, k)
            )
            grad_hat = [weights_grad, torch.clone(values.grad)]
            for g1, g2 in zip(grad, grad_hat):
                self.assertLess(
                    torch.abs(g1 - g2).max(),
                    1e-3
                )

    def test_correctness(self):
        N = 12
        H = 6
        L = 1000
        S = 1000
        E = 32
        k = 32
        C = 100
        I = 10
        B = 32
        for exp in range(30):
            N = np.random.randint(1, 6)
            H = np.random.randint(1, 8)
            C = np.random.randint(10, 500)
            L = np.random.randint(C, 2000)
            E = np.random.randint(10, 128)
            S = np.random.randint(100, 1000)
            k = np.random.randint(10, 64)

            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L S E C k: "
                       "{} {} {} {} {} {} {}").format(N, H, L, S, E, C, k))

            Q = torch.randn(N, H, L, E).to(self.device)
            K = torch.randn(N, H, S, E).to(self.device)
            lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
            groups, counts = cluster_queries(Q, lengths, C, I, B)

            sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)
            q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)
            s_queries = Q.reshape(-1, E).index_select(
                0, q_flat).view(N, H, L, E)

            Q_grouped = aggregate(
                s_queries, sorted_g.view(N, H, L), 1/counts.float()
            )

            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
            _, topk = torch.topk(QK, k, dim=-1)
            topk = topk.contiguous()
            topk_broadcast = broadcast(
                topk.float(),
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, k), device=Q.device)
            )
            weights_sorted = torch.rand(
                N, H, L, k).to(self.device).requires_grad_(True)
            weights_sorted.retain_grad()

            q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
            weights = torch.clone(
                weights_sorted.reshape(-1, k).index_select(
                    0, q_rev_flat).view(N, H, L, k)
            )
            weights.retain_grad()

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

            self._zero_grad(weights_sorted, values)
            self._zero_grad(weights, values)

            output_hat_sorted = clustered_sparse_weighted_average(
                weights_sorted, values, topk, sorted_g.view(N, H, L), counts
            )
            output_hat = output_hat_sorted.reshape(-1, E).index_select(
                0, q_rev_flat).view(N, H, L, E)

            self.assertLess(
                torch.abs(output - output_hat).max(),
                1e-4
            )
            output_hat.sum().backward()
            weights_grad_sorted = torch.clone(weights_sorted.grad)
            weights_grad = torch.clone(
                weights_grad_sorted.reshape(-1, k).index_select(
                    0, q_rev_flat).view(N, H, L, k)
            )
            grad_hat = [weights_grad, torch.clone(values.grad)]
            for g1, g2 in zip(grad, grad_hat):
                self.assertLess(
                    torch.abs(g1 - g2).max(),
                    1e-3
                )

    def test_forward(self):
        N = 12
        H = 5
        L = 100
        S = 100
        E = 32
        C = 10
        I = 10
        B = 32
        k = 5

        for exp in range(30):
            C = np.random.randint(10, 500)
            L = np.random.randint(C, 2000)
            E = np.random.randint(10, 128)
            S = np.random.randint(100, 1000)
            k = np.random.randint(10, 64)

            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L S E C k: "
                       "{} {} {} {} {} {} {}").format(N, H, L, S, E, C, k))

            Q = torch.randn(N, H, L, E).to(self.device)
            K = torch.randn(N, H, S, E).to(self.device)
            lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
            groups, counts = cluster_queries(Q, lengths, C, I, B)

            sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)
            q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)
            s_queries = Q.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)

            Q_grouped = aggregate(
                s_queries, sorted_g.view(N, H, L), 1/counts.float()
            )

            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
            _, topk = torch.topk(QK, k, dim=-1)
            topk = topk.contiguous()
            topk_broadcast = broadcast(
                topk.float(),
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, k), device=Q.device)
            )

            weights_sorted = clustered_sparse_dot_product(
                s_queries, K, topk, sorted_g.view(N, H, L), counts, lengths
            )
            weights = torch.softmax(weights_sorted, dim=-1)

            q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
            weights = weights_sorted.reshape(-1, k).index_select(
                0, q_rev_flat).view(N, H, L, k)
            values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
            values_selected = values[
                torch.arange(N).view(N, 1, 1, 1).to(self.device),
                torch.arange(H).view(1, H, 1, 1).to(self.device),
                topk_broadcast.long()
            ]

            output = (weights.unsqueeze(-1)*values_selected).sum(-2)
            output_hat_sorted = clustered_sparse_weighted_average(
                weights_sorted, values, topk, sorted_g.view(N, H, L), counts
            )
            output_hat = output_hat_sorted.reshape(-1, E).index_select(
                0, q_rev_flat).view(N, H, L, E)

            self.assertLess(
                torch.abs(output_hat - output).max(),
                1e-3
            )


if __name__ == "__main__":
    unittest.main()
