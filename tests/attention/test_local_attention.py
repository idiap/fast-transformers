#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
#


import os
import time
import unittest

import torch

from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.attention.full_attention import FullAttention
from fast_transformers.attention.local_attention import LocalAttention


class TestLocalAttention(unittest.TestCase):
    def _get_inputs(self, N=10, L=5, S=8, H=4, E=32, D=64, device="cpu"):
        return (
            torch.rand(N, L, H, E).to(device),
            torch.rand(N, S, H, E).to(device),
            torch.rand(N, S, H, D).to(device),
            FullMask(L, S, device=device),
            FullMask(N, L, device=device),
            FullMask(N, S, device=device)
        )

    def test_forward(self):
        att = LocalAttention(3, softmax_temp=1)
        q, k, v, m1, m2, m3 = self._get_inputs()
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

    def test_masked(self):
        att = LocalAttention(16, softmax_temp=1)
        q, k, v, m1, m2, m3 = self._get_inputs(N=3, L=64, S=64, D=32)
        m2 = m3 = LengthMask(torch.tensor([8, 16, 64], dtype=torch.long))
        v_hat = att(q, k, v, m1, m2, m3)
        self.assertFalse(torch.any(torch.isnan(v_hat)))

    def test_compare_with_full(self):
        local_att = LocalAttention(17, softmax_temp=1).eval()
        full_att = FullAttention(softmax_temp=1).eval()

        q, k, v, m1, m2, m3 = self._get_inputs(N=10, L=128, S=128, D=32)
        m = FullMask(
            torch.abs(torch.arange(128)[:, None] - torch.arange(128)[None]) < 9
        )
        v_full = full_att(q, k, v, m, m2, m3)
        v_local = local_att(q, k, v, m1, m2, m3)

        self.assertTrue(torch.allclose(v_full, v_local, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_cpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=1024, S=1024, E=64, D=64)
        att = LocalAttention(128)

        # warmup the cache
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)

        # measure
        start = time.time()
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)
        end = time.time()
        print("CPU Time taken:", (end-start)*1000, "(ms)")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA capable device")
    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_gpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=1024, S=1024, E=64, D=64,
                                               device="cuda")
        att = LocalAttention(128)

        # warmup the caches
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)

        # measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)
        end.record()
        torch.cuda.synchronize()
        print("GPU time taken:", start.elapsed_time(end), "(ms)")


if __name__ == "__main__":
    unittest.main()

