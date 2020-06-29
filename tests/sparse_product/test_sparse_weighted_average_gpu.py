#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import os
from os import getenv
import unittest

import torch

from fast_transformers.sparse_product import sparse_weighted_average


class TestSparseWeightedAverage(unittest.TestCase):
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

    def test_correctness(self):
        N = 2
        H = 4
        L = 3000
        S = 3000
        E = 32
        k = 32
        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)

        attn = torch.randn(N, H, L, S).to(self.device).requires_grad_(False) 
        topk_v, topk = torch.topk(attn, k, dim=-1)

        self._zero_grad(weights, values)
        values_selected = values[
            torch.arange(N).view(N, 1, 1, 1).to(self.device),
            torch.arange(H).view(1, H, 1, 1).to(self.device),
            topk
        ]
        output = (weights.unsqueeze(-1)*values_selected).sum(-2)
        output.sum().backward()
        grad = [torch.clone(weights.grad), torch.clone(values.grad)]

        self._zero_grad(weights, values)
        output_hat = sparse_weighted_average(weights, values, topk)
        output_hat.sum().backward()
        grad_hat = [torch.clone(weights.grad), torch.clone(values.grad)]
        self.assertLess(
            torch.abs(output - output_hat).max(),
            1e-4
        )
        for g1, g2 in zip(grad, grad_hat):
            self.assertLess(
                torch.abs(g1 - g2).max(),
                1e-4
            )

    def test_forward(self):
        N = 5
        H = 2 
        L = 100
        S = 100
        E = 32
        k = 5

        weights = torch.arange(0,k).expand(N, H, L, k).to(self.device).float().requires_grad_(True)
        values = torch.arange(0,E).expand(N, H, L, E).to(self.device).float().requires_grad_(True)

        attn = torch.arange(0, S).expand(N, H, L, S).to(self.device).float().requires_grad_(False)
        topk_v, topk = torch.topk(attn, k, dim=-1)

        values_selected = values[
            torch.arange(N).view(N, 1, 1, 1).to(self.device),
            torch.arange(H).view(1, H, 1, 1).to(self.device),
            topk
        ]
        output = (weights.unsqueeze(-1)*values_selected).sum(-2)
        output_hat = sparse_weighted_average(weights, values, topk)
        self.assertLess(
            torch.abs(output - output_hat).max(),
            1e-4
        )

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward(self):
        N = 12
        H = 8
        L = 2000
        S = 2000
        E = 32
        k = 32

        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)
        
        attn = torch.randn(N, H, L, S).to(self.device).requires_grad_(False) 
        topk_v, topk = torch.topk(attn, k, dim=-1)
        topk = topk.contiguous()
        for i in range(2000):
            output_hat = sparse_weighted_average(weights, values, topk)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        output_hat = sparse_weighted_average(weights, values, topk)
        e.record()
        torch.cuda.synchronize()
        t_sparse = s.elapsed_time(e)

        print('T_sparse Forward:{}'.format(t_sparse))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_backward(self):
        N = 12
        H = 8
        L = 2000
        S = 2000
        E = 32
        k = 32

        weights = torch.rand(N, H, L, k).to(self.device).requires_grad_(True)
        values = torch.randn(N, H, S, E).to(self.device).requires_grad_(True)

        attn = torch.randn(N, H, L, S).to(self.device).requires_grad_(False)
        topk_v, topk = torch.topk(attn, k, dim=-1)
        topk = topk.contiguous()
        for i in range(2000):
            output_hat = sparse_weighted_average(weights, values, topk)

        self._zero_grad(weights, values)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        output_hat.sum().backward()
        e.record()
        torch.cuda.synchronize()
        t_sparse = s.elapsed_time(e)

        print('T_sparse Backward:{}'.format(t_sparse))

if __name__ == "__main__":
    unittest.main()
