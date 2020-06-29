#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import time

import torch

from fast_transformers.hashing import hash_cpu


def simple_lsh(X, A):
    B = (torch.einsum("nj,ij->ni", [X, A[:, :-1]]) > A[None, :, -1]).long()
    bits = 2**torch.arange(A.shape[0])
    return torch.einsum("ni,i->n", [B, bits])


class TestHashCPU(unittest.TestCase):
    def test_hash(self):
        for bits in range(10, 63):
            x = torch.rand(100, 32)
            a = torch.randn(bits, 33)
            a[:,-1] = 0.0
            h1 = simple_lsh(x, a)
            h2 = torch.zeros_like(h1)
            h3 = torch.zeros_like(h1)
            hash_cpu.compute_hashes(x, a, h2)
            self.assertTrue(torch.all(h1==h2))
            B = torch.einsum("nj,ij->ni", [x, a[:, :-1]])
            hash_cpu.compute_hashes_from_projections(B, h3)
            self.assertTrue(torch.all(h1==h3))

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS", ""), "no benchmarks")
    def test_benchmark_hash(self):
        N = 12
        L = 1000
        H = 8
        E = 32
        B = 63
        x = torch.rand(N*L*H, E)
        a = torch.randn(B, E+1)
        a[:,-1] = 0.
        h1 = simple_lsh(x, a)
        h2 = torch.zeros_like(h1)
        h3 = torch.zeros_like(h1)

        # Count simple pytorch
        for i in range(50):
            simple_lsh(x, a)
        t = time.time()
        for i in range(50):
            simple_lsh(x, a)
        d1 = time.time()-t

        # Count simple C++ pytorch
        for i in range(50):
            hash_cpu.compute_hashes(x, a, h2)
        t = time.time()
        for i in range(50):
            hash_cpu.compute_hashes(x, a, h2)
        d2 = time.time()-t

        # Count simple C++ pytorch version 2
        for i in range(50):
            P = torch.einsum("nj,ij->ni", [x, a[:, :-1]])
            hash_cpu.compute_hashes_from_projections(P, h3)
        t = time.time()
        for i in range(50):
            P = torch.einsum("nj,ij->ni", [x, a[:, :-1]])
            hash_cpu.compute_hashes_from_projections(P, h3)
        d3 = time.time()-t

        print(d1, d2, d3, d1/d2, d2/d3, d1/d3)


if __name__ == "__main__":
    unittest.main()
