#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import os
import time
import unittest

import torch

from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.attention.aft_attention import AFTFullAttention, \
    AFTSimpleAttention


class TestAFTAttention(unittest.TestCase):
    def _get_inputs(self, N=10, L=5, S=8, H=4, E=32, D=32, device="cpu"):
        return (
            torch.rand(N, L, H, E).to(device),
            torch.rand(N, S, H, E).to(device),
            torch.rand(N, S, H, D).to(device),
            FullMask(L, S, device=device),
            FullMask(N, L, device=device),
            FullMask(N, S, device=device)
        )

    def test_forward(self):
        att = AFTFullAttention()
        q, k, v, m1, m2, m3 = self._get_inputs()
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

        att = AFTSimpleAttention()
        q, k, v, m1, m2, m3 = self._get_inputs()
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

    def test_masking(self):
        q, k, v, m1, m2, m3 = self._get_inputs()
        m1 = FullMask(torch.rand(5, 8) > 0.5)

        att = AFTFullAttention()
        v = att(q, k, v, m1, m2, m3)

        att = AFTSimpleAttention()
        with self.assertRaises(ValueError):
            v = att(q, k, v, m1, m2, m3)

        q, k, v, m1, m2, m3 = self._get_inputs(L=8, S=8)
        m1 = FullMask(torch.tril(torch.ones(8, 8, dtype=torch.bool)))
        v = att(q, k, v, m1, m2, m3)

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_cpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=256, S=256, E=64, D=64)
        att_full = AFTFullAttention()
        att_simple = AFTSimpleAttention()

        for name, att in zip(["full", "simple"], [att_full, att_simple]):
            # warmup the cache
            for i in range(10):
                v_new = att(q, k, v, m1, m2, m3)

            # measure
            start = time.time()
            for i in range(10):
                v_new = att(q, k, v, m1, m2, m3)
            end = time.time()
            print("AFT", name, "CPU Time taken:", (end-start)*1000, "(ms)")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA capable device")
    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_gpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=256, S=256, E=64, D=64,
                                               device="cuda")
        att_full = AFTFullAttention().cuda()
        att_simple = AFTSimpleAttention()

        for name, att in zip(["full", "simple"], [att_full, att_simple]):
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
            print("AFT", name, "GPU time taken:", start.elapsed_time(end), "(ms)")


if __name__ == "__main__":
    unittest.main()
