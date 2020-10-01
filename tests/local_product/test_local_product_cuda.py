#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import os
import time
import unittest

import numpy as np
import torch

try:
    from fast_transformers.local_product.local_product_cuda import \
        local_dot_product, local_dot_backward, \
        local_weighted_average, local_weighted_average_backward
except ImportError:
    local_dot_product = None
    local_dot_backward = None
    local_weighted_average = None
    local_weighted_average_backward = None


class TestLocalProductCUDA(unittest.TestCase):
    kernels = {
        "normal": {
            "dot": local_dot_product,
            "dot_backward": local_dot_backward,
            "wa": local_weighted_average,
            "wa_backward": local_weighted_average_backward
        },
        # map the optimized versions to other keys here
    }

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA available")

    def _test_result_forward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            local_context = np.random.randint(8, 24)
            mask = torch.zeros(L, L).cuda()
            lengths = torch.full((N,), L, dtype=torch.long).cuda()
            out = self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)

            QK = torch.full((N, H, L, local_context), -1e24,
                            dtype=torch.float32).cuda()
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

    def _test_benchmark_forward(self, CP):
        N = 10
        L = 2048
        H = 12
        E = 64
        Q = torch.rand(N, H, L, E).cuda()
        K = torch.rand(N, H, L, E).cuda()
        local_context = 512
        mask = torch.zeros(L, L).cuda()
        lengths = torch.full((N,), L, dtype=torch.long).cuda()

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)

        # measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)
        end.record()
        torch.cuda.synchronize()
        print("[dot] [{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            torch.einsum("nhle,nhse->nhls", Q, K) + mask
        end.record()
        torch.cuda.synchronize()
        print("[full_dot] [{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_forward(k)

    def _test_result_backward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            local_context = np.random.randint(8, 24)
            lengths = torch.full((N,), L, dtype=torch.long).cuda()
            grad_in = torch.ones(N, H, L, local_context).cuda()
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

    def _test_benchmark_backward(self, CP):
        N = 10
        L = 2048
        H = 12
        E = 64
        Q = torch.rand(N, H, L, E).cuda()
        K = torch.rand(N, H, L, E).cuda()
        local_context = 512
        mask = torch.zeros(L, L).cuda()
        lengths = torch.full((N,), L, dtype=torch.long).cuda()
        grad_in = torch.ones(N, H, L, local_context).cuda()

        # warmup the cache
        for i in range(10):
            GQ, GK = self.kernels[CP]["dot_backward"](Q, K, lengths, grad_in,
                                                        local_context)

        # measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            GQ, GK = self.kernels[CP]["dot_backward"](Q, K, lengths, grad_in,
                                                        local_context)
        end.record()
        torch.cuda.synchronize()
        print("[dot_backward] [{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_backward(k)

    def _test_result_weighted_average(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            local_context = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1).cuda()
            V = torch.rand(N, H, L, E).cuda()
            out_hat = self.kernels[CP]["wa"](A, V)

            out = torch.zeros(N, H, L, E).cuda()
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

    def _test_benchmark_weighted_average(self, CP):
        N = 10
        L = 4096
        H = 12
        E = 64
        Q = torch.rand(N, H, L, E).cuda()
        K = torch.rand(N, H, L, E).cuda()
        local_context = 512
        A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1).cuda()
        V = torch.rand(N, H, L, E).cuda()

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["wa"](A, V)

        # measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            self.kernels[CP]["wa"](A, V)
        end.record()
        torch.cuda.synchronize()
        print("[wa] [{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_weighted_average(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_weighted_average(k)

    def _test_result_weighted_average_backward(self, CP):
        for t in range(10):
            N = 10
            L = 100
            H = 10
            E = np.random.randint(10, 256)
            local_context = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1).cuda()
            V = torch.rand(N, H, L, E).cuda()
            grad_in = torch.ones(N, H, L, E).cuda()
            GA, GV = self.kernels[CP]["wa_backward"](A, V, grad_in)

            A = A.requires_grad_(True)
            V = V.requires_grad_(True)
            out = torch.zeros(N, H, L, E).cuda()
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

    def _test_benchmark_weighted_average_backward(self, CP):
        N = 10
        L = 4096
        H = 12
        E = 64
        Q = torch.rand(N, H, L, E).cuda()
        K = torch.rand(N, H, L, E).cuda()
        local_context = 512
        A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1).cuda()
        V = torch.rand(N, H, L, E).cuda()
        grad_in = torch.ones(N, H, L, E).cuda()

        # warmup the cache
        for i in range(10):
            self.kernels[CP]["wa_backward"](A, V, grad_in)

        # measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            self.kernels[CP]["wa_backward"](A, V, grad_in)
        end.record()
        torch.cuda.synchronize()
        print("[wa_back] [{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_weighted_average_backward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_weighted_average_backward(k)


if __name__ == "__main__":
    unittest.main()
