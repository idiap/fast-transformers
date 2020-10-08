#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import unittest

import torch


from fast_transformers.masking import TriangularCausalMask, FullMask
from fast_transformers.attention import CausalLinearAttention


class TestCausalLinearAttention(unittest.TestCase):
    def _get_inputs(self, N=10, L=5, S=8, H=4, E=32, D=64, device="cpu"):
        return (
            torch.rand(N, L, H, E).to(device),
            torch.rand(N, S, H, E).to(device),
            torch.rand(N, S, H, D).to(device),
            TriangularCausalMask(L, device=device),
            FullMask(N, L, device=device),
            FullMask(N, S, device=device)
        )

    def test_forward(self):
        att = CausalLinearAttention(32)
        q, k, v, m1, m2, m3 = self._get_inputs(L=5, S=5)
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

        q, k, v, m1, m2, m3 = self._get_inputs(L=5, S=10)
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())

        q, k, v, m1, m2, m3 = self._get_inputs(L=10, S=5)
        v = att(q, k, v, m1, m2, m3)
        self.assertTrue(v.is_contiguous())


if __name__ == "__main__":
    unittest.main()
