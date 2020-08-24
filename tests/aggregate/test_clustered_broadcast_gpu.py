#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch
from torch.nn.init import normal_
from torch.nn import functional as F

from fast_transformers.hashing import compute_hashes
from fast_transformers.clustering.hamming import cluster
from fast_transformers.aggregate import aggregate, broadcast
from fast_transformers.aggregate import clustered_broadcast

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


class TestClusteredBroadcastGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_broadcast(self):
        N = 10
        H = 3
        L = 500
        S = 500
        E = 32
        C = 4
        I = 5
        B = 16

        Q = torch.randn(N, H, L, E).cuda()
        lengths = torch.full((N,), L, dtype=torch.int32).cuda()
        lengths[1] = 400
        lengths[3] = 200
        lengths[7] = 450
        lengths[8] = 150
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        K = torch.randn(N, H, S, E).cuda()
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)

        V = torch.randn(N, H, S, E).cuda()
        A = F.softmax(QK, dim=-1)
        V_new = torch.einsum("nhls,nhse->nhle", A, V)
        V_broadcast = torch.zeros((N, H, L, E), dtype=V_new.dtype).cuda()
        #V_broadcast = broadcast_clustered(V_new, groups, counts, lengths, V_broadcast)
        V_broadcast = clustered_broadcast(V_new, groups, counts, lengths, V_broadcast)

        V_broadcast_2 = broadcast(
            V_new,
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, E), device=Q.device)
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
        E = 32
        C = 200
        I = 5
        B = 32

        Q = torch.randn(N, H, L, E).cuda()
        lengths = torch.full((N,), L, dtype=torch.int32).cuda()
        groups, counts = cluster_queries(Q, lengths, C, I, B)
        Q_grouped = aggregate(Q, groups, 1/counts.float())
        K = torch.randn(N, H, S, E).cuda()
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, K)

        V = torch.randn(N, H, S, E).cuda()
        A = F.softmax(QK, dim=-1)
        V_new = torch.einsum("nhls,nhse->nhle", A, V)
        V_broadcast = torch.zeros((N, H, L, E), dtype=V_new.dtype).cuda()
        #V_broadcast = broadcast_clustered(V_new, groups, counts, lengths, V_broadcast)
        V_broadcast = clustered_broadcast(V_new, groups, counts, lengths, V_broadcast)

        V_broadcast_2 = broadcast(
            V_new,
            groups,
            torch.ones_like(counts, dtype=torch.float32),
            torch.zeros((N, H, L, E), device=Q.device)
        )

        self.assertLess(
            torch.max(torch.abs(
                V_broadcast_2
                - V_broadcast
                )
            ),
            1e-4
        )

        for i in range(2000):
            #V_broadcast = broadcast_clustered(V_new, groups, counts, lengths, V_broadcast)
            V_broadcast = clustered_broadcast(V_new, groups, counts, lengths, V_broadcast)


        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        V_broadcast = clustered_broadcast(V_new, groups, counts, lengths, V_broadcast)
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
