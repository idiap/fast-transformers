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


class TestSparseProductCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_single_query(self):
        X = torch.randn(1, 1, 1, 32).cuda()
        Y = torch.randn(1, 1, 100, 32).cuda()
        lengths = torch.full((1,), 1, dtype=torch.int32).cuda()
        topk = (torch.cumsum(torch.rand(1, 1, 1, 10)*10, dim=-1)).long().cuda()

        products = sparse_dot_product(
            X,
            Y,
            topk,
        )
        all_products = torch.einsum("nhle,nhse->nhls", X, Y)

        self.assertLess(
            torch.max(torch.abs(
                products.squeeze() -
                all_products[0, 0, 0, topk[0, 0, 0]]
           )),
            1e-4
        )

    def test_simple_product(self):
        X = torch.randn(10, 4, 100, 32).cuda()
        Y = torch.randn(10, 4, 100, 32).cuda()
        lengths = torch.full((10,), 100, dtype=torch.int32).cuda()
        topk = (torch.cumsum(torch.rand(10, 4, 100, 10)*10, dim=-1)).long().cuda()

        A = torch.randn(10, 4, 100, 100).to(X.device).requires_grad_(False) 
        topk_v, topk = torch.topk(A, 10, dim=-1)
        topk = topk.contiguous()

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
        X = torch.randn(N, H, L, E).cuda()
        Y = torch.randn(N, H, S, E).cuda()

        A = torch.randn(N, H, L, S).to(X.device).requires_grad_(False) 
        topk_v, topk = torch.topk(A, k, dim=-1)
        topk = topk.contiguous()

        for i in range(1000):
            products = sparse_dot_product(
                X,
                Y,
                topk,
            )
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        products = sparse_dot_product(
            X,
            Y,
            topk,
        )
        e.record()
        torch.cuda.synchronize()
        t_s = s.elapsed_time(e)
        for i in range(1000):
            torch.einsum("nhle,nhse->nhls", X, Y)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.einsum("nhle,nhse->nhls", X, Y)
        e.record()
        torch.cuda.synchronize()
        t_f = s.elapsed_time(e)
        print("Sparse: {}, Full: {}, F/S: {}".format(t_s, t_f, t_f/t_s))


if __name__ == "__main__":
    unittest.main()
