#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

from fast_transformers.attention.attention_layer import AttentionLayer


class TestAttentionLayer(unittest.TestCase):
    def _assert_sizes_attention(self, qshape, kshape, vshape):
        def inner(q, k, v, m1, m2, m3):
            self.assertEqual(q.shape, qshape)
            self.assertEqual(k.shape, kshape)
            self.assertEqual(v.shape, vshape)
            N, L, H, E = q.shape
            _, S, _, D = v.shape
            return v.new_zeros((N, L, H, D))
        return inner

    def test_forward(self):
        att = AttentionLayer(
            self._assert_sizes_attention(
                (10, 5, 4, 25),
                (10, 8, 4, 25),
                (10, 8, 4, 25)
            ),
            100,
            4
        )
        v = att(
            torch.rand(10, 5, 100),
            torch.rand(10, 8, 100),
            torch.rand(10, 8, 100),
            None, None, None
        )
        self.assertEqual(v.shape, (10, 5, 100))

        att = AttentionLayer(
            self._assert_sizes_attention(
                (10, 5, 4, 32),
                (10, 8, 4, 32),
                (10, 8, 4, 64)
            ),
            100,
            4,
            d_keys=32,
            d_values=64
        )
        v = att(
            torch.rand(10, 5, 100),
            torch.rand(10, 8, 100),
            torch.rand(10, 8, 100),
            None, None, None
        )
        self.assertEqual(v.shape, (10, 5, 100))


if __name__ == "__main__":
    unittest.main()
