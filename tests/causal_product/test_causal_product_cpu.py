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

from fast_transformers.causal_product.causal_product_cpu import \
    causal_dot_product, causal_dot_backward


def max_relative_error(a, b, eps=1e-6):
    return torch.abs((a-b) / (torch.abs(a) + eps)).max().item()


class TestCausalProductCPU(unittest.TestCase):
    kernels = {
        "normal": {
            "forward": causal_dot_product,
            "backward": causal_dot_backward
        },
        # map the optimized versions to other keys here
    }

    def _test_result_forward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            M = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)
            V = torch.rand(N, H, L, M)
            out = torch.zeros(N, H, L, M)
            self.kernels[CP]["forward"](Q, K, V, out)

            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            mask = torch.tril(torch.ones(L, L))
            out2 = torch.einsum("nhls,nhsm,ls->nhlm", QK, V, mask)

            self.assertLess(max_relative_error(out2, out), 1e-5)

    def _test_result_backward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            M = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)
            V = torch.rand(N, H, L, M)
            go = torch.rand(N, H, L, M)
            gq, gk, gv = [torch.zeros_like(x) for x in [Q, K, V]]
            self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)

            Q, K, V = [x.requires_grad_(True) for x in [Q, K, V]]
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            mask = torch.tril(torch.ones(L, L))
            out = torch.einsum("nhls,nhsm,ls->nhlm", QK, V, mask)
            l = torch.einsum("nhlm,nhlm->", out, go)
            l.backward()

            self.assertLess(max_relative_error(Q.grad, gq), 1e-5)
            self.assertLess(max_relative_error(K.grad, gk), 1e-5)
            self.assertLess(max_relative_error(V.grad, gv), 1e-5)

    def _test_benchmark_forward(self, CP):
        N = 10
        L = 1000
        H = 10
        E = 32
        M = 64
        Q = torch.rand(N, H, L, E)
        K = torch.rand(N, H, L, E)
        V = torch.rand(N, H, L, M)
        out = torch.rand(N, H, L, M)

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["forward"](Q, K, V, out)

        # measure
        start = time.time()
        for i in range(10):
            self.kernels[CP]["forward"](Q, K, V, out)
        end = time.time()
        print("[{}] CPU time taken: {} (ms)".format(
            CP,
            (end-start)*1000
        ))

    def _test_benchmark_backward(self, CP):
        N = 10
        L = 1000
        H = 10
        E = 32
        M = 64
        Q = torch.rand(N, H, L, E)
        K = torch.rand(N, H, L, E)
        V = torch.rand(N, H, L, M)
        go = torch.rand(N, H, L, M)
        gq, gk, gv = [torch.zeros_like(x) for x in [Q, K, V]]

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)

        # measure
        start = time.time()
        for i in range(10):
            self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)
        end = time.time()
        print("[{}] CPU time taken: {} (ms)".format(
            CP,
            (end-start)*1000
        ))

    def test_result_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_forward(k)

    def test_result_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_backward(k)

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_forward(k)

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_backward(k)


if __name__ == "__main__":
    unittest.main()
