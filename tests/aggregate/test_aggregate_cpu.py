#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch

from fast_transformers.aggregate import aggregate_cpu, broadcast_cpu


class TestAggregateCPU(unittest.TestCase):
    def test_aggregate(self):
        N = 1
        H = 1
        L = 40
        E = 2
        C = 4

        x = torch.rand((N, H, L, E))
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int()
        c = (.1*torch.ones(N, H, C))
        y = torch.zeros(N, H, C, E)

        aggregate_cpu(x, g, c, y)
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

        y = torch.rand(N, H, C, E)
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int()
        c = (.1*torch.ones(N, H, C))
        x = torch.rand((N, H, L, E))

        broadcast_cpu(y, g, c, x)
        for i in range(C):
            self.assertTrue(
                torch.all(x[:, :, i::C] == 0.1*y[:, :, i:i+1, :])
            )

    def test_both(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        x_start = torch.rand(N, H, L, E)
        x_end = torch.rand(N, H, L, E)
        g = (torch.rand(N, H, L)*C).int()
        c = torch.zeros(N, H, C)
        y = torch.zeros((N, H, C, E))

        # Aggregating ones should give us the counts
        aggregate_cpu(
            torch.ones(N, H, L, 1),
            g,
            torch.ones(N, H, C),
            c
        )
        for i in range(C):
            self.assertTrue(torch.all((g == i).sum(2) == c[:, :, i].long()))

        # Aggregating into averages twice should be a noop
        aggregate_cpu(x_start, g, 1/c, y)
        broadcast_cpu(y, g, torch.ones(N, H, C), x_start)
        y.zero_()
        aggregate_cpu(x_start, g, 1/c, y)
        broadcast_cpu(y, g, torch.ones(N, H, C), x_end)
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

        x = torch.rand((N, H, L, E))
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int()
        g[:, :, -4:] = -1
        c = torch.ones(N, H, C)/9.
        y = torch.zeros(N, H, C, E)

        aggregate_cpu(x, g, c, y)
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
        x = torch.rand((N, H, L, E))
        g = (torch.arange(L) % C).view(L, 1, 1).repeat(1, N, H).int()
        c = 0.1*torch.ones(C, N, H)
        y = torch.zeros((C, N, H, E))

        s = time.time()
        for i in range(100):
            aggregate_cpu(x, g, c, y)
        e = time.time()
        t_aggregate = e - s

        print('Aggregate Time: {}'.format(t_aggregate))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_broadcast_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 32
        C = 100

        y = torch.rand((N, H, C, E))
        g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int()
        c = 0.1*torch.ones(N, H, C)
        x = torch.zeros((N, H, L, E))

        s = time.time()
        for i in range(100):
            broadcast_cpu(y, g, c, x)
        e = time.time()
        t_broadcast = e - s

        print('Broadcast Time: {}'.format(t_broadcast))


if __name__ == "__main__":
    unittest.main()
