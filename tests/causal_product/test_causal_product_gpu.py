#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import os
import unittest

import numpy as np
import torch

try:
    from fast_transformers.causal_product.causal_product_cuda import \
        causal_dot_product, causal_dot_backward

    # import optimized versions here
except ImportError:
    causal_dot_product = causal_dot_backward = None
    # define them as None here so that the class can be declared


def max_relative_error(a, b, eps=1e-6):
    return torch.abs((a-b) / (torch.abs(a) + eps)).max().item()


class TestCausalProductCUDA(unittest.TestCase):
    kernels = {
        "normal": {
            "forward": causal_dot_product,
            "backward": causal_dot_backward
        },
        # map the optimized versions to other keys here
    }

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA available")

    def _test_result_forward(self, CP):
        for t in range(20):
            N = 4
            L = 100
            H = 10
            E = np.random.randint(10, 512)
            M = np.random.randint(10, 512)
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            V = torch.rand(N, H, L, M).cuda()
            out = torch.zeros(N, H, L, M).cuda()
            self.kernels[CP]["forward"](Q, K, V, out)

            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            mask = torch.tril(torch.ones(L, L)).cuda()
            out2 = torch.einsum("nhls,nhsm,ls->nhlm", QK, V, mask)

            self.assertLess(max_relative_error(out2, out), 1e-5)

    def _test_result_backward(self, CP):
        for t in range(20):
            N = 4
            L = 100
            H = 10
            E = np.random.randint(10, 512)
            M = np.random.randint(10, 512)
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            V = torch.rand(N, H, L, M).cuda()
            go = torch.rand(N, H, L, M).cuda()
            gq, gk, gv = [torch.zeros_like(x) for x in [Q, K, V]]
            self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)

            Q, K, V = [x.requires_grad_(True) for x in [Q, K, V]]
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            mask = torch.tril(torch.ones(L, L)).cuda()
            out = torch.einsum("nhls,nhsm,ls->nhlm", QK, V, mask)
            l = torch.einsum("nhlm,nhlm->", out, go)
            l.backward()

            self.assertLess(max_relative_error(Q.grad, gq), 1e-5)
            self.assertLess(max_relative_error(K.grad, gk), 1e-5)
            self.assertLess(max_relative_error(V.grad, gv), 1e-5)

    def _test_benchmark_forward(self, CP):
        print("{:>4} {:>5} {:>5} {:>5} {:>5} {:>12}".format("N", "L", "H", "E", "M", "FW Time (ms)"))
        for N, L, H, E, M in [
            [8, 4096, 8, 512, 512],
            [1, 4096, 8, 512, 512],
            [16, 512, 8, 512, 512],
            [16, 128, 8, 512, 512],
            [1, 128, 8, 512, 512],
            [16, 4096, 1, 512, 512],
            [16, 4096, 8, 64, 64],
        ]:
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            V = torch.rand(N, H, L, M).cuda()
            out = torch.rand(N, H, L, M).cuda()

            # warmup the cache
            for i in range(10):
                self.kernels[CP]["forward"](Q, K, V, out)

            # measure
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(10):
                self.kernels[CP]["forward"](Q, K, V, out)
            end.record()
            torch.cuda.synchronize()
            print("{:>5} {:>5} {:>5} {:>5} {:>5} {:>8.1f}".format(
                N, L, H, E, M,
                start.elapsed_time(end)
            ))

    def _test_benchmark_backward(self, CP):
        print("{:>4} {:>5} {:>5} {:>5} {:>5} {:>12}".format("N", "L", "H", "E", "M", "BW Time (ms)"))
        for N, L, H, E, M in [
            [8, 4096, 8, 512, 512],
            [1, 4096, 8, 512, 512],
            [16, 512, 8, 512, 512],
            [16, 128, 8, 512, 512],
            [1, 128, 8, 512, 512],
            [16, 4096, 1, 512, 512],
            [16, 4096, 8, 64, 64],
        ]:
            Q = torch.rand(N, H, L, E).cuda()
            K = torch.rand(N, H, L, E).cuda()
            V = torch.rand(N, H, L, M).cuda()
            go = torch.rand(N, H, L, M).cuda()
            gq, gk, gv = [torch.zeros_like(x) for x in [Q, K, V]]

            # warmup the cache
            for i in range(10):
                self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)

            # measure
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(10):
                self.kernels[CP]["backward"](Q, K, V, go, gq, gk, gv)
            end.record()
            torch.cuda.synchronize()
            print("{:>5} {:>5} {:>5} {:>5} {:>5} {:>8.1f}".format(
                N, L, H, E, M,
                start.elapsed_time(end)
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

