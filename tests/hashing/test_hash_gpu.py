#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch

try:
    from fast_transformers.hashing import hash_cuda
except ImportError:
    pass

def simple_lsh(X, A):
    X = X.cpu()
    A = A.cpu()
    B = (torch.einsum("nj,ij->ni", [X, A[:, :-1]]) > A[None, :, -1]).long()
    bits = 2**torch.arange(A.shape[0])
    return torch.einsum("ni,i->n", [B, bits]).cuda()


class TestHashGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_hash(self):
        for bits in range(10, 63):
            x = torch.rand(100, 32).to("cuda")
            a = torch.randn(bits, 33).to("cuda")
            a[:,-1] = 0.0
            h1 = simple_lsh(x, a)
            h2 = torch.zeros_like(h1)
            h3 = torch.zeros_like(h1)
            hash_cuda.compute_hashes(x, a, h2)
            self.assertTrue(torch.all(h1==h2))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_hash(self):
        N = 12
        L = 1000
        H = 8
        E = 32
        B = 63
        x = torch.rand(N*L*H, 32).to("cuda")
        a = torch.randn(B, 33).to("cuda")
        h1 = simple_lsh(x, a)
        h2 = torch.zeros_like(h1)
        h3 = torch.zeros_like(h1)

        # Count simple pytorch
        for i in range(50):
            simple_lsh(x, a)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        simple_lsh(x, a)
        e.record()
        torch.cuda.synchronize()
        t_simple = s.elapsed_time(e)

        # Count simple C++ pytorch
        for i in range(50):
            hash_cuda.compute_hashes(x, a, h2)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        hash_cuda.compute_hashes(x, a, h2)
        e.record()
        torch.cuda.synchronize()
        t_cuda = s.elapsed_time(e)

        print(t_simple, t_cuda, t_simple/t_cuda)



if __name__ == "__main__":
    unittest.main()

