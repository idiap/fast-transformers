#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import os
import time
import unittest

import numpy as np
import torch

from fast_transformers.local_product.local_product_cuda import \
    local_dot_product, local_weighted_average


class TestLocalProductCUDA(unittest.TestCase):
    kernels = {
        "normal": {
            "dot": local_dot_product,
            "wa": local_weighted_average
        },
        # map the optimized versions to other keys here
    }

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA available")

    def _test_result_forward(self, CP):
        # N = 1
        # L = 10
        # H = 1
        # E = 32
        # Q = torch.rand(N, H, L, E).cuda()
        # K = torch.rand(N, H, L, E).cuda()
        # local_context = 8
        # mask = torch.zeros(L, L).cuda()
        # lengths = torch.full((N,), L, dtype=torch.long).cuda()
        # out = self.kernels[CP]["dot"](Q, K, mask, lengths, local_context)

        # QK = torch.full((N, H, L, local_context), float("-inf"),
        #                 dtype=torch.float32).cuda()
        # for i in range(L):
        #     start = i - local_context//2
        #     end = start + local_context
        #     start = max(0, start)
        #     end = min(L, end)
        #     kstart = local_context//2 - abs(i-start)
        #     QK[:, :, i, kstart:kstart+(end-start)] = torch.einsum(
        #         "nhe,nhle->nhl",
        #         Q[:, :, i],
        #         K[:, :, start:end]
        #     )
        # diff = torch.abs(out-QK)
        # print(out[0, 0])

        # self.assertTrue(torch.allclose(QK, out, atol=1e-5, rtol=1e-5))

        # return

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

            QK = torch.full((N, H, L, local_context), float("-inf"),
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
        print("[{}] GPU time taken: {} (ms)".format(
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
        print("[{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_forward(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_forward(k)

    def _test_result_weighted_average(self, CP):
        #N = 1
        #L = 10
        #H = 1
        #E = 32
        #local_context = 8
        #A = torch.softmax(torch.randn(N, H, L, local_context), dim=-1).cuda()
        #V = torch.rand(N, H, L, E).cuda()
        #out_hat = self.kernels[CP]["wa"](A, V)

        #out = torch.zeros(N, H, L, E).cuda()
        #for i in range(L):
        #    start = i - local_context//2
        #    end = start + local_context
        #    start = max(0, start)
        #    end = min(L, end)
        #    kstart = local_context//2 - abs(i-start)
        #    out[:, :, i] = torch.einsum(
        #        "nhl,nhle->nhe",
        #        A[:, :, i, kstart:kstart+end-start],
        #        V[:, :, start:end]
        #    )
        #diff = out_hat - out

        #self.assertTrue(torch.allclose(out, out_hat, atol=1e-5, rtol=1e-5))
        #return

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
        L = 2048
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
        print("[{}] GPU time taken: {} (ms)".format(
            CP,
            start.elapsed_time(end)
        ))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_weighted_average(self):
        for k in self.kernels.keys():
            with self.subTest(msg=k):
                self._test_benchmark_weighted_average(k)


if __name__ == "__main__":
    unittest.main()
