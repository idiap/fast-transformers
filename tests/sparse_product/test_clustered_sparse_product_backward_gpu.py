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


def sparse_product(Q, K, groups, topk, counts, lengths):
    N, H, L, E = Q.shape
    _, _, C, k = topk.shape

    sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
    sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

    q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
    q_flat = (sorted_gi + q_offset).reshape(-1)

    # sorted queries, keys, values
    s_queries = Q.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
    Q_grouped = aggregate(s_queries, sorted_g.view(N, H, L), 1/counts.float())
    topk = topk.contiguous()

    products_sorted = clustered_sparse_dot_product(
        s_queries, K, topk, sorted_g.view(N, H, L), counts, lengths
    )
    q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
    products = products_sorted.reshape(-1, k).index_select(
        0, q_rev_flat).view(N, H, L, k)
    return products


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

    def test_masked_simple_grad(self):
        N = 4
        H = 2
        L = 100
        E = 64
        S = 100
        k = 32
        C = 5
        I = 5
        B = 16

        for i in range(30):
            C = np.random.randint(10, 500)
            L = np.random.randint(C, 2000)
            E = np.random.randint(10, 128)
            S = np.random.randint(100, 1000)
            k = np.random.randint(10, 64)

            if os.getenv("VEROSE_TESTS", ""):
                print(("Testing Masked: N H L S E C k: "
                       "{} {} {} {} {} {} {}").format(N, H, L, S, E, C, k))

            Q = torch.randn(N, H, L, E).to(self.device).requires_grad_(True)
            K = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)

            lengths = np.random.randint(C, L+1, N)
            lengths = torch.tensor(lengths, dtype=torch.int32).to(self.device)
            query_lengths = LengthMask(
                lengths,
                max_len=L
            )
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

            QK_selected = QK_selected * query_lengths.float_matrix[:, None, :, None]
            QK_selected.sum().backward()
            grad = [torch.clone(Q.grad), torch.clone(K.grad)]

            self._zero_grad(Q, K)
            QK_selected_hat = sparse_product(
                Q, K, groups, topk, counts, lengths
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
                    1e-3
                )

    def test_simple_grad(self):
        N = 2
        H = 2
        L = 100
        E = 64
        S = 100
        k = 32
        C = 5
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
        QK_selected_hat = sparse_product(Q, K, groups, topk, counts, lengths)
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

    def test_difficult_grad(self):
        N = 12
        H = 5
        I = 5
        B = 16

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
            Q.requires_grad = True
            K.requires_grad = True
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
            QK_selected_hat = sparse_product(
                Q, K, groups, topk, counts, lengths
            )
            QK_selected_hat.sum().backward()
            grad_hat = [torch.clone(Q.grad), torch.clone(K.grad)]

            self.assertLess(
                torch.abs(QK_selected - QK_selected_hat).max(),
                1e-4
            )
            i = 0
            for g1, g2 in zip(grad, grad_hat):
                self.assertLess(
                    torch.abs(g1 - g2).max(),
                    1e-3
                )
                i += 1

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_backward(self):
        N = 12
        H = 8
        L = 1024
        S = 1024
        E = 64
        k = 32
        C = 100
        I = 10
        B = 32

        Q = torch.randn(N, H, L, E).to(self.device).requires_grad_(True)
        K = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)

        self._zero_grad(Q, K)
        for i in range(100):
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
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
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
        sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

        q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
        q_flat = (sorted_gi + q_offset).reshape(-1)

        s_queries = Q.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)

        Q_grouped = aggregate(Q, groups, 1/counts.float())
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
        _, topk = torch.topk(QK, k, dim=-1)
        topk = topk.contiguous()
        products_sorted = clustered_sparse_dot_product(
            s_queries,
            K,
            topk,
            groups,
            counts,
            lengths
        )
        q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
        products = products_sorted.reshape(-1, k).index_select(
            0, q_rev_flat).view(N, H, L, k)

        for i in range(100):
            QK = clustered_sparse_dot_product(
                s_queries, K, topk,
                groups, counts,
                lengths
            )
            QK = QK.reshape(-1, k).index_select(0, q_rev_flat).view(N, H, L, k)
        self._zero_grad(Q, K)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        QK = clustered_sparse_dot_product(
            Q, K, topk,
            groups, counts,
            lengths
        )
        QK = QK.reshape(-1, k).index_select(0, q_rev_flat).view(N, H, L, k)
        s.record()
        QK.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_sparse = s.elapsed_time(e)
        print("Benchmark Backward: T_Full: {}, T_Sparse: {}".format(
            t_full, t_sparse))


if __name__ == "__main__":
    unittest.main()
