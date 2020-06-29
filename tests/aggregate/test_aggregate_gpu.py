#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch

try:
    from fast_transformers.aggregate import aggregate_gpu, broadcast_gpu
except ImportError:
    pass


class TestAggregateGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_aggregate(self):
        N = 1
        H = 1
        L = 40
        E = 2
        C = 4

        x = torch.rand((N, H, L, E)).cuda()
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
        c = (.1*torch.ones(N, H, C)).cuda()
        y = torch.zeros(N, H, C, E).cuda()

        aggregate_gpu(x, g, c, y)
        for i in range(C):
            self.assertLess(
                torch.abs(
                    x[:, :, i::C, :].mean(2) - y[:, :, i, :]
                ).max().item(),
                1e-6
            )

    def test_broadcast(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        y = torch.rand(N, H, C, E).cuda()
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
        c = (.1*torch.ones(N, H, C)).cuda()
        x = torch.rand((N, H, L, E)).cuda()

        broadcast_gpu(y, g, c, x)
        for i in range(C):
            self.assertTrue(
                torch.all(
                    x[:, :, i::C] == 0.1*y[:, :, i:i+1, :]
                )
            )

    def test_both(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        x_start = torch.rand(N, H, L, E).cuda()
        x_end = torch.rand(N, H, L, E).cuda()
        g = (torch.rand(N, H, L)*C).int().cuda()
        c = torch.zeros(N, H, C).cuda()
        y = torch.zeros((N, H, C, E)).cuda()

        # Aggregating ones should give us the counts
        aggregate_gpu(
            torch.ones(N, H, L, 1).cuda(),
            g,
            torch.ones(N, H, C).cuda(),
            c.unsqueeze(-1)
        )
        for i in range(C):
            self.assertTrue(
                torch.all(
                    (g == i).sum(2) == c[:, :, i].long()
                )
            )

        c = c.view(N, H, C)
        # Aggregating into averages twice should be a noop
        aggregate_gpu(x_start, g, 1./c, y)
        broadcast_gpu(y, g, torch.ones(N, H, C).cuda(), x_start)
        y.zero_()
        aggregate_gpu(x_start, g, 1./c, y)
        broadcast_gpu(y, g, torch.ones(N, H, C).cuda(), x_end)
        self.assertLess(
            torch.abs(x_start-x_end).max().item(),
            1e-6
        )

    def test_aggregate_masked(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        x = torch.rand((N, H, L, E)).cuda()
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
        g[:, :, -4:] = -1
        c = torch.ones(N, H, C).cuda()/9.
        y = torch.zeros(N, H, C, E).cuda()

        aggregate_gpu(x, g, c, y)
        for i in range(C):
            self.assertLess(
                torch.abs(
                    x[:, :, i::C, :][:, :, :-1, :].mean(2) - y[:, :, i, :]
                ).max().item(),
                1e-6
            )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_aggregate_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 32
        C = 100
        x = torch.rand(L, N, H, E).cuda()
        g = (torch.arange(L) % C).view(L, 1, 1).repeat(1, N, H).int().cuda()
        c = (0.1*torch.ones(C, N, H)).cuda()
        y = torch.zeros((C, N, H, E)).cuda()

        for i in range(2000):
            aggregate_gpu(x, g, c, y)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        aggregate_gpu(x, g, c, y)
        e.record()
        torch.cuda.synchronize()
        t_aggregate = s.elapsed_time(e)

        print('Aggregate Time: {}'.format(t_aggregate))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_broadcast_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 32
        C = 100

        y = torch.rand(N, H, C, E).cuda()
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cuda()
        c = (.1*torch.ones(N, H, C)).cuda()
        x = torch.zeros((N, H, L, E)).cuda()

        for i in range(2000):
            broadcast_gpu(y, g, c, x)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        broadcast_gpu(y, g, c, x)
        e.record()
        torch.cuda.synchronize()
        t_broadcast = s.elapsed_time(e)

        print('Broadcast Time: {}'.format(t_broadcast))


if __name__ == "__main__":
    unittest.main()
