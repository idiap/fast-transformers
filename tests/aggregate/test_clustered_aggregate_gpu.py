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

try:
    from fast_transformers.aggregate import clustered_aggregate, \
            clustered_broadcast
except ImportError:
    pass


class TestAggregateGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_aggregate(self):
        N = 2
        H = 4
        L = 80
        E = 2
        C = 4

        for i in range(30):
            C = np.random.randint(5, 10)
            L = np.random.randint(1, 30) * C
            E = np.random.randint(10, 128)
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L E C: "
                       "{} {} {} {} {}").format(N, H, L, E, C))

            x = torch.rand((N, H, L, E)).cuda()
            g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
            f = torch.ones(N, H, C).cuda() * (C / L)
            counts = torch.ones_like(f, dtype=torch.int32) * (L // C)
            y = torch.zeros(N, H, C, E).cuda()
            lengths = torch.full((N,), L, dtype=torch.int32).to(x.device)

            sorted_g, sorted_gi = torch.sort(g.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

            q_offset = torch.arange(N*H, device=x.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)

            # sorted queries, keys, values
            s_x = x.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
            y = clustered_aggregate(
                s_x, sorted_g.view(N, H, -1), f, lengths, y
            )
            for i in range(C):
                self.assertLess(
                    torch.abs(
                        x[:, :, i::C, :].mean(2) - y[:, :, i, :]
                    ).max().item(),
                    1e-6
                )

    def test_aggregate_masked(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        for i in range(30):
            C = np.random.randint(5, 10)
            L = np.random.randint(2, 30) * C
            E = np.random.randint(10, 128)
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L E C: "
                       "{} {} {} {} {}").format(N, H, L, E, C))
            x = torch.rand((N, H, L, E)).cuda()
            g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
            g[:, :, -C:] = C + 1
            c = (L // C) - 1

            lengths = torch.full((N,), L-C, dtype=torch.int32).to(x.device)
            f = torch.ones(N, H, C).cuda() / float(c)
            counts = torch.ones_like(f, dtype=torch.int32) * c
            y = torch.zeros(N, H, C, E).cuda()

            sorted_g, sorted_gi = torch.sort(g.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

            q_offset = torch.arange(N*H, device=x.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)

            # sorted queries, keys, values
            s_x = x.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
            y = clustered_aggregate(
                s_x, sorted_g.view(N, H, -1), f, lengths, y
            )

            for i in range(C):
                x_m = x[:, :, i::C, :][:, :, :-1, :].mean(2)
                self.assertLess(
                    torch.abs(
                        x_m - y[:, :, i, :]
                    ).max().item(),
                    1e-6
                )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_aggregate_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 64
        C = 200
        x = torch.rand(N, H, L, E).cuda()
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
        lengths = torch.full((N,), L, dtype=torch.int32).to(x.device)
        c = (0.1*torch.ones(N, H, C)).cuda()
        y = torch.zeros((N, H, C, E)).cuda()

        for i in range(2000):
            clustered_aggregate(x, g, c, lengths, y)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        clustered_aggregate(x, g, c, lengths, y)
        e.record()
        torch.cuda.synchronize()
        t_aggregate = s.elapsed_time(e)

        print('Aggregate Time: {}'.format(t_aggregate))

    # @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    # def test_broadcast_benchmark(self):
    #     N = 12
    #     H = 8
    #     L = 1000
    #     S = 1000
    #     E = 64
    #     C = 200

    #     y = torch.rand(N, H, C, E).cuda()
    #     g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
    #     c = (.1*torch.ones(N, H, C)).cuda()
    #     x = torch.zeros((N, H, L, E)).cuda()

    #     for i in range(2000):
    #         broadcast_gpu(y, g, c, x)

    #     s = torch.cuda.Event(enable_timing=True)
    #     e = torch.cuda.Event(enable_timing=True)
    #     s.record()
    #     broadcast_gpu(y, g, c, x)
    #     e.record()
    #     torch.cuda.synchronize()
    #     t_broadcast = s.elapsed_time(e)

    #     print('Broadcast Time: {}'.format(t_broadcast))


if __name__ == "__main__":
    unittest.main()
