#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import os
import time
import unittest

import torch

from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.attention.linear_attention import LinearAttention


class TestLinearAttention(unittest.TestCase):
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
        att = LinearAttention(32)
        q, k, v, m1, m2, m3 = self._get_inputs()
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

    def test_masking(self):
        att = LinearAttention(32)
        q, k, v, m1, m2, m3 = self._get_inputs()

        # Make sure that we raise an error if m1 is not all ones
        with self.assertRaises(RuntimeError):
            att(q, k, v, FullMask(torch.rand(*m1.shape) > 0.5), m2, m3)

        # Make sure that the key lengths is paid attention to
        q, k, v, m1, m2, m3 = self._get_inputs(S=10, D=1)
        m3 = LengthMask(torch.tensor(list(range(10)))+1)
        for i in range(9):
            v[i, i+1:] = 1e9
        v_new = att(q, k, v, m1, m2, m3)
        self.assertLess(v_new.max().item(), 1)


    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_cpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=1024, S=1024, E=64, D=64)
        att = LinearAttention(64)

        # warmup the cache
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)

        # measure
        start = time.time()
        for i in range(10):
            v_new = att(q, k, v, m1, m2, m3)
        end = time.time()
        print("CPU time taken:", (end-start)*1000, "(ms)")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA capable device")
    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_gpu(self):
        q, k, v, m1, m2, m3 = self._get_inputs(L=1024, S=1024, E=64, D=64,
                                               device="cuda")
        att = LinearAttention(64)

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
