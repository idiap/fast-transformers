#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import numpy as np
import time

import torch
from torch.nn.init import normal_
from torch.nn import functional as F

from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.aggregate import clustered_broadcast, \
        clustered_aggregate


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


def try_sorted_broadcast(Q, K, V, groups, counts, lengths, Q_grouped_orig):
    N, H, L, E = Q.shape
    _, _, S, D = V.shape
    sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
    sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

    q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
    q_flat = (sorted_gi + q_offset).reshape(-1)

    # sorted queries, keys, values
    s_queries = Q.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
    Q_grouped = clustered_aggregate(
        s_queries, sorted_g.view(N, H, L), 1 / counts.float(), lengths
    )
    assert(abs(Q_grouped_orig - Q_grouped).max().item() < 1e-4)
    QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)
    A = F.softmax(QK, dim=-1)
    V_new = torch.einsum("nhls,nhse->nhle", A, V)
    V_broadcast = torch.zeros((N, H, L, E), dtype=V_new.dtype).cuda()
    factors = torch.ones_like(counts, dtype=torch.float32)
    V_sorted_broadcast = clustered_broadcast(
        V_new, sorted_g.view(N, H, L), counts, factors, V_broadcast
    )

    q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
    V_broadcast_remap = V_sorted_broadcast.reshape(-1, D).index_select(
        0, q_rev_flat).view(N, H, L, D)
    return V_broadcast_remap


class TestClusteredBroadcastGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_broadcast_full(self):
        N = 3
        H = 2
        L = 400
        S = 100
        E = 256
        C = 211
        I = 5
        B = 16

        for exp in range(50):
            Q = torch.randn(N, H, L, E).cuda()
            lengths = torch.full((N,), L, dtype=torch.int32).cuda()
            lengths[0] = np.random.randint(C, L+1)
            lengths[1] = np.random.randint(C, L+1)
            groups, counts = cluster_queries(Q, lengths, C, I, B)
            Q_grouped = aggregate(Q, groups, 1/counts.float())
            K = torch.randn(N, H, S, E).cuda()
            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)

            V = torch.randn(N, H, S, E).cuda()
            A = F.softmax(QK, dim=-1)
            V_new = torch.einsum("nhls,nhse->nhle", A, V)
            V_broadcast_2 = broadcast(
                V_new,
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, E), device=Q.device)
            )
            V_broadcast = try_sorted_broadcast(
                Q, K, V, groups, counts, lengths, Q_grouped
            )
            self.assertLess(
                torch.max(torch.abs(
                    V_broadcast_2
                    - V_broadcast
                    )
                ),
                1e-4
            )

    def test_broadcast_difficult(self):
        N = 10
        H = 3
        E = 64
        I = 5
        B = 16

        for exp in range(20):
            S = np.random.randint(100, 1000)
            C = np.random.randint(10, 500)
            L = np.random.randint(C, 2000)
            E = np.random.randint(10, 160)
            lengths = torch.tensor(
                np.random.randint(C, L+1, N),
                dtype=torch.int32
            ).cuda()
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Test: N H L S E C: "
                       "{} {} {} {} {} {}").format(N, H, L, S, E, C))

            Q = torch.randn(N, H, L, E).cuda()
            groups, counts = cluster_queries(Q, lengths, C, I, B)
            Q_grouped = aggregate(Q, groups, 1/counts.float())
            K = torch.randn(N, H, S, E).cuda()
            QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)

            V = torch.randn(N, H, S, E).cuda()
            A = F.softmax(QK, dim=-1)
            V_new = torch.einsum("nhls,nhse->nhle", A, V)
            V_broadcast_2 = broadcast(
                V_new,
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, E), device=Q.device)
            )
            V_broadcast = try_sorted_broadcast(
                Q, K, V, groups, counts, lengths, Q_grouped
            )
            self.assertLess(
                torch.max(torch.abs(
                    V_broadcast_2
                    - V_broadcast
                    )
                ),
                1e-4
            )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_broadcast_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 64
        D = 64
        C = 200
        I = 5
        B = 63

        Q = torch.randn(N, H, L, E).cuda()
        lengths = torch.full((N,), L, dtype=torch.int32).cuda()
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        sorted_g, sorted_gi = torch.sort(groups.view(N*H, -1), dim=-1)
        sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

        q_offset = torch.arange(N*H, device=Q.device).unsqueeze(-1) * L
        q_flat = (sorted_gi + q_offset).reshape(-1)

        Q_grouped = aggregate(Q, groups, 1/counts.float())
        K = torch.randn(N, H, S, E).cuda()
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)

        V = torch.randn(N, H, S, E).cuda()
        A = F.softmax(QK, dim=-1)
        V_new = torch.einsum("nhls,nhse->nhle", A, V)
        V_broadcast = torch.zeros((N, H, L, E), dtype=V_new.dtype).cuda()
        factors = torch.ones_like(counts, dtype=torch.float32)
        V_sorted_broadcast = clustered_broadcast(
            V_new, sorted_g.view(N, H, L), counts, factors, V_broadcast
        )
        q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
        V_broadcast = V_sorted_broadcast.reshape(-1, D).index_select(
                0, q_rev_flat).view(N, H, L, D)

        for i in range(2000):
            factors = torch.ones_like(counts, dtype=torch.float32)
            V_sorted_broadcast = clustered_broadcast(
                V_new, sorted_g.view(N, H, L), counts, factors, V_broadcast
            )
            q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
            V_broadcast = V_sorted_broadcast.reshape(-1, D).index_select(
                    0, q_rev_flat).view(N, H, L, D)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        factors = torch.ones_like(counts, dtype=torch.float32)
        V_sorted_broadcast = clustered_broadcast(
            V_new, sorted_g.view(N, H, L), counts, factors, V_broadcast
        )
        q_rev_flat = (sorted_rev_gi + q_offset).reshape(-1)
        V_broadcast = V_sorted_broadcast.reshape(-1, D).index_select(
                0, q_rev_flat).view(N, H, L, D)
        e.record()
        torch.cuda.synchronize()
        t_broadcast = s.elapsed_time(e)

        for i in range(200):
            V_broadcast_2 = broadcast(
                V_new,
                groups,
                torch.ones_like(counts, dtype=torch.float32),
                torch.zeros((N, H, L, E), device=Q.device)
            )

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        V_broadcast_2 = broadcast(
            V_new,
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, E), device=Q.device)
        )
        e.record()
        torch.cuda.synchronize()
        t_broadcast_2 = s.elapsed_time(e)

        print("B1: {}, B2: {}".format(t_broadcast, t_broadcast_2))


if __name__ == "__main__":
    unittest.main()
