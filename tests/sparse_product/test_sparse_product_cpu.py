#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
import time
import unittest

import torch

from fast_transformers.sparse_product import sparse_dot_product


class TestSparseProductCPU(unittest.TestCase):
    def test_simple_product(self):
        X = torch.randn(10, 4, 100, 32)
        Y = torch.randn(10, 4, 100, 32)
        topk = (torch.cumsum(torch.rand(10, 4, 100, 10)*10, dim=-1)).long()

        products = sparse_dot_product(
            X,
            Y,
            topk,
        )

        all_products = torch.einsum("nhle,nhse->nhls", X, Y)
        self.assertLess(
            torch.max(torch.abs(
                products -
                all_products[
                    torch.arange(10).view(10, 1, 1, 1),
                    torch.arange(4).view(1, 4, 1, 1),
                    torch.arange(100).view(1, 1, 100, 1),
                    topk
                ]
            )),
            1e-4
        )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_small_benchmark(self):
        N = 12
        H = 8
        L = 1000
        S = 1000
        E = 32
        k = 32
        X = torch.randn(N, H, L, E)
        Y = torch.randn(N, H, S, E)
        topk = (torch.cumsum(torch.rand(N, H, L, k)*40, dim=-1)).long()

        n_runs = 10
        s = time.time()
        for run in range(n_runs):
            products = sparse_dot_product(
                X,
                Y,
                topk,
            )
        e = time.time()
        t_s = (e - s) / n_runs

        s = time.time()
        for run in range(n_runs):
            torch.einsum("nhle,nhse->nhls", X, Y)
        e = time.time()
        t_f = (e - s) / n_runs
        print("Sparse: {}, Full: {}, F/S: {}".format(t_s, t_f, t_f/t_s))


if __name__ == "__main__":
    unittest.main()
