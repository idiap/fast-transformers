#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import os
import time
import unittest

import numpy as np
import torch

from fast_transformers.local_product.local_product_cpu import \
    local_dot_product, local_dot_backward, \
    local_weighted_average, local_weighted_average_backward


class TestLocalProductCPU(unittest.TestCase):
    kernels = {
        "normal": {
            "dot": local_dot_product,
            "dot_backward": local_dot_backward,
            "wa": local_weighted_average,
            "wa_backward": local_weighted_average_backward
        },
        # map the optimized versions to other keys here
    }

    def _test_result_forward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)
            local_context = np.random.randint(8, 24)
            mask = torch.zeros(L, L)
            lengths = torch.full((N,), L, dtype=torch.long)
            out = self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)

            QK = torch.full((N, H, L, local_context), -1e24,
                            dtype=torch.float32)
            for i in range(L):
                start = i - local_context//2
                end = start + local_context
                start = max(0, start)
                end = min(L, end)
                kstart = local_context//2 - abs(i-start)
                QK[:, :, i, kstart:kstart+(end-start)] = torch.einsum(
                    "nhe,nhle->nhl",
                    Q[:, :, i],
                    K[:, :, start:end]
                )

            self.assertTrue(torch.allclose(QK, out, atol=1e-5, rtol=1e-5))

    def test_result_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_forward(k)

    def _test_result_backward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)
            local_context = np.random.randint(8, 24)
            lengths = torch.full((N,), L, dtype=torch.long)
            grad_in = torch.ones(N, H, L, local_context)
            GQ, GK = self.kernels[CP]["dot_backward"](Q, K, lengths, grad_in,
                                                        local_context)

            Q = Q.requires_grad_(True)
            K = K.requires_grad_(True)
            QK = torch.full((N, H, L, local_context), -1e24,
                            dtype=torch.float32)
            for i in range(L):
                start = i - local_context//2
                end = start + local_context
                start = max(0, start)
                end = min(L, end)
                kstart = local_context//2 - abs(i-start)
                QK[:, :, i, kstart:kstart+(end-start)] = torch.einsum(
                    "nhe,nhle->nhl",
                    Q[:, :, i],
                    K[:, :, start:end]
                )
            QK.sum().backward()

            self.assertTrue(torch.allclose(Q.grad, GQ, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(K.grad, GK, atol=1e-5, rtol=1e-5))

    def test_result_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_backward(k)

    def _test_benchmark_forward(self, CP):
        N = 10
        L = 2048
        H = 12
        E = 64
        Q = torch.rand(N, H, L, E)
        K = torch.rand(N, H, L, E)
        local_context = 512
        mask = torch.zeros(L, L)
        lengths = torch.full((N,), L, dtype=torch.long)

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)

        # measure
        start = time.time()
        for i in range(10):
            self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)
        end = time.time()
        print("[{}] CPU time taken: {} (ms)".format(
            CP,
            (end-start)*1000
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_forward(k)

    def _test_result_weighted_average(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            local_context = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1)
            V = torch.rand(N, H, L, E)
            out_hat = self.kernels[CP]["wa"](A, V)

            out = torch.zeros(N, H, L, E)
            for i in range(L):
                start = i - local_context//2
                end = start + local_context
                start = max(0, start)
                end = min(L, end)
                kstart = local_context//2 - abs(i-start)
                out[:, :, i] = torch.einsum(
                    "nhl,nhle->nhe",
                    A[:, :, i, kstart:kstart+end-start],
                    V[:, :, start:end]
                )

            self.assertTrue(torch.allclose(out, out_hat, atol=1e-5, rtol=1e-5))

    def test_result_weighted_average(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_weighted_average(k)

    def _test_result_weighted_average_backward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            local_context = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1)
            V = torch.rand(N, H, L, E)
            grad_in = torch.ones(N, H, L, E)
            GA, GV = self.kernels[CP]["wa_backward"](A, V, grad_in)

            A = A.requires_grad_(True)
            V = V.requires_grad_(True)
            out = torch.zeros(N, H, L, E)
            for i in range(L):
                start = i - local_context//2
                end = start + local_context
                start = max(0, start)
                end = min(L, end)
                kstart = local_context//2 - abs(i-start)
                out[:, :, i] = torch.einsum(
                    "nhl,nhle->nhe",
                    A[:, :, i, kstart:kstart+end-start],
                    V[:, :, start:end]
                )
            out.sum().backward()

            self.assertTrue(torch.allclose(A.grad, GA, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(V.grad, GV, atol=1e-5, rtol=1e-5))

    def test_result_weighted_average_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_result_weighted_average_backward(k)


if __name__ == "__main__":
    unittest.main()
