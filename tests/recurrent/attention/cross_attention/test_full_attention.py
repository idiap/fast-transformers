#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import os
import time
import unittest

import torch

from fast_transformers.attention import FullAttention
from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.recurrent.attention import RecurrentCrossFullAttention


class TestRecurrentCrossFullAttention(unittest.TestCase):
    def test_correctness(self):
        # Prepare the inputs
        N = 10
        H = 4
        E = 25
        M = 64
        L = 42
        S = 100
        q = torch.rand(N, L, H, E)
        k = torch.rand(N, S, H, E)
        v = torch.rand(N, S, H, M)
        m1 = FullMask(L, S)
        m2 = LengthMask(torch.full((N,), L, dtype=torch.int64))
        m3 = LengthMask(torch.full((N,), S, dtype=torch.int64))

        # Get the outputs from the attention in batch mode
        att = FullAttention()
        att.eval()
        v_out1 = att(q, k, v, m1, m2, m3)

        # Get the output from the attention in recurrent mode
        att = RecurrentCrossFullAttention()
        att.eval()
        v_out2_unstacked = []
        state = None
        for i in range(L):
            vi, state = att(q[:, i], k, v, m3, state=state)
            v_out2_unstacked.append(vi)
        v_out2 = torch.stack(v_out2_unstacked, dim=1)

        # Check that they match
        self.assertLess(torch.abs(v_out1 - v_out2).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
