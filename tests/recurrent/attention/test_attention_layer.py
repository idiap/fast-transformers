#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

from fast_transformers.recurrent.attention import RecurrentAttentionLayer


class TestRecurrentAttentionLayer(unittest.TestCase):
    def _assert_sizes_attention(self, qshape, kshape, vshape):
        def inner(q, k, v, m):
            self.assertEqual(q.shape, qshape)
            self.assertEqual(k.shape, kshape)
            self.assertEqual(v.shape, vshape)
            N, H, E = q.shape
            _, _, D = v.shape
            return v.new_zeros((N, H, D)), m
        return inner

    def test_forward(self):
        att = RecurrentAttentionLayer(
            self._assert_sizes_attention(
                (10, 4, 25),
                (10, 4, 25),
                (10, 4, 25)
            ),
            100,
            4
        )
        v, m = att(
            torch.rand(10, 100),
            torch.rand(10, 100),
            torch.rand(10, 100),
            "test memory"
        )
        self.assertEqual(v.shape, (10, 100))
        self.assertEqual(m, "test memory")

        att = RecurrentAttentionLayer(
            self._assert_sizes_attention(
                (10, 4, 32),
                (10, 4, 32),
                (10, 4, 64)
            ),
            100,
            4,
            d_keys=32,
            d_values=64
        )
        v, m = att(
            torch.rand(10, 100),
            torch.rand(10, 100),
            torch.rand(10, 100),
            "test memory"
        )
        self.assertEqual(v.shape, (10, 100))
        self.assertEqual(m, "test memory")


if __name__ == "__main__":
    unittest.main()
