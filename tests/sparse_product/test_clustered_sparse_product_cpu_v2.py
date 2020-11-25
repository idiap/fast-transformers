#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
import time
import unittest

import numpy as np

import torch
from torch.nn.init import normal_

from fast_transformers.aggregate import aggregate, broadcast, \
        clustered_aggregate
from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.sparse_product import sparse_dot_product
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


def sparse_product(Q, K, groups, topk, counts, lengths, k, Q_grouped_orig):
    N, H, L, E = Q.shape
    sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
    sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

    q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
    q_flat = (sorted_gi + q_offset).reshape(-1)

    # rearrage queries
    s_queries = Q.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
    Q_grouped = clustered_aggregate(
        s_queries, sorted_g.view(N, H, L), 1/counts.float(), lengths
    )
    topk = topk.contiguous()

    products_sorted = clustered_sparse_dot_product(
        s_queries, K, topk, sorted_g.view(N, H, L), counts, lengths
    )
    q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
    products = products_sorted.reshape(-1, k).index_select(0, q_rev_flat)
    products = products.view(N, H, L, k)

    return products, Q_grouped


class TestSparseProductCUDA(unittest.TestCase):
    @property
    def device(self):
        return "cpu"

    def test_simple_product(self):
        N = 2
        H = 2
        L = 100
        E = 32
        S = 50
        k = 32
        C = 5
        I = 5
        B = 16

        for i in range(20):
            k = np.random.randint(10, S)
            E = np.random.randint(10, 129)
            k = 32
            E = 32
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L S E C k: "
                       "{} {} {} {} {} {} {}").format(N, H, L, S, E, C, k))

            Q = torch.randn(N, H, L, E).to(self.device)
            K = torch.randn(N, H, S, E).to(self.device)
            lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
            lengths[1] = 50
            lengths[1] = 45
            lengths[1] = 10
            groups, counts = cluster_queries(Q, lengths, C, I, B)
            Q_grouped = aggregate(Q, groups, 1/counts.float())
            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
            _, topk = torch.topk(QK, k, sorted=False, dim=-1)
            topk = topk.contiguous()
            products, Q_grouped_alt = sparse_product(
                Q, K, groups, topk, counts, lengths, k, Q_grouped
            )
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
            for i in range(N):
                p_1 = products[i, :, :lengths[i], :]
                p_2 = products_2[i, :, :lengths[i], :]
                self.assertLess(
                    torch.max(torch.abs(
                        p_2 - p_1
                        )
                    ),
                    1e-4
                )

    def test_difficult_product(self):
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
            lengths = torch.full((N,), L, dtype=torch.int32).to(self.device)
            groups, counts = cluster_queries(Q, lengths, C, I, B)

            Q_grouped = aggregate(Q, groups, 1/counts.float())
            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
            _, topk = torch.topk(QK, k, dim=-1)
            topk = topk.contiguous()

            products, _ = sparse_product(
                Q, K, groups, topk, counts, lengths, k, Q_grouped
            )
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


if __name__ == "__main__":
    unittest.main()
