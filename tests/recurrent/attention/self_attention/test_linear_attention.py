#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import os
import time
import unittest

import torch

from fast_transformers.attention import CausalLinearAttention
from fast_transformers.masking import TriangularCausalMask, LengthMask
from fast_transformers.recurrent.attention import RecurrentLinearAttention


class TestRecurrentLinearAttention(unittest.TestCase):
    def test_forward(self):
        # Prepare the inputs
        N = 10
        H = 4
        E = 25
        M = 64
        L = 100
        q = torch.rand(N, H, E)
        k = torch.rand(N, H, E)
        v = torch.rand(N, H, M)
        memory = [
            torch.rand(N, H, E, M),
            torch.rand(N, H, E)
        ]

        # Test the attention module
        att = RecurrentLinearAttention()
        v_new, mem_new = att(q, k, v)
        self.assertEqual(v_new.shape, (N, H, M))
        self.assertEqual(len(mem_new), 2)
        self.assertEqual(mem_new[0].shape, (N, H, E, M))
        self.assertEqual(mem_new[1].shape, (N, H, E))
        v_new, mem_new = att(q, k, v, mem_new)
        self.assertEqual(v_new.shape, (N, H, M))
        self.assertEqual(len(mem_new), 2)
        self.assertEqual(mem_new[0].shape, (N, H, E, M))
        self.assertEqual(mem_new[1].shape, (N, H, E))

        v_new, mem_new = att(q, k, v, memory)
        self.assertEqual(v_new.shape, (N, H, M))
        self.assertEqual(len(mem_new), 2)
        self.assertEqual(mem_new[0].shape, (N, H, E, M))
        self.assertEqual(mem_new[1].shape, (N, H, E))

    def test_correctness(self):
        # Prepare the inputs
        N = 10
        H = 4
        E = 25
        M = 64
        L = 100
        q = torch.rand(N, L, H, E)
        k = torch.rand(N, L, H, E)
        v = torch.rand(N, L, H, M)
        m1 = TriangularCausalMask(L)
        m2 = LengthMask(torch.full((N,), L, dtype=torch.long))
        m3 = LengthMask(torch.full((N,), L, dtype=torch.long))
        att = CausalLinearAttention()
        rec_att = RecurrentLinearAttention()
        att.eval()
        rec_att.eval()

        v1 = att(q, k, v, m1, m2, m3)
        v2 = []
        memory = None
        for i in range(L):
            v2i, memory = rec_att(q[:, i], k[:, i], v[:, i], memory)
            v2.append(v2i)
        v2 = torch.stack(v2, dim=1)
        self.assertLess(torch.abs(v1-v2).max(), 1e-5)

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_cpu(self):
        # Prepare the inputs
        N = 10
        H = 12
        E = 25
        M = 64
        L = 100
        q = torch.rand(N, H, E)
        k = torch.rand(N, H, E)
        v = torch.rand(N, H, M)
        memory = None
        att = RecurrentLinearAttention()

        start = time.time()
        for i in range(100):
            v, memory = att(q, k, v, memory)
        end = time.time()
        print("CPU Time taken:", (end-start)*1000, "(ms)")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA capable device")
    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_gpu(self):
        # Prepare the inputs
        N = 10
        H = 12
        E = 25
        M = 64
        L = 100
        q = torch.rand(N, H, E).cuda()
        k = torch.rand(N, H, E).cuda()
        v = torch.rand(N, H, M).cuda()
        memory = None
        att = RecurrentLinearAttention()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(100):
            v, memory = att(q, k, v, memory)
        end.record()
        torch.cuda.synchronize()
        print("GPU time taken:", start.elapsed_time(end), "(ms)")


if __name__ == "__main__":
    unittest.main()

