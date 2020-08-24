#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import unittest

import torch

from fast_transformers.recurrent.attention import RecurrentCrossAttentionLayer


class TestRecurrentCrossAttentionLayer(unittest.TestCase):
    def _assert_sizes_attention(self, qshape, kshape, vshape):
        def inner(q, k, v, kl, state=None):
            if state is not None:
                k, v = state
            self.assertEqual(q.shape, qshape)
            self.assertEqual(k.shape, kshape)
            self.assertEqual(v.shape, vshape)
            N, H, E = q.shape
            _, _, _, D = v.shape
            return v.new_zeros((N, H, D)), [k, v]
        return inner

    def test_forward(self):
        att = RecurrentCrossAttentionLayer(
            self._assert_sizes_attention(
                (10, 4, 25),
                (10, 42, 4, 25),
                (10, 42, 4, 25)
            ),
            100,
            4
        )

        v, s = att(
            torch.rand(10, 100),
            torch.rand(10, 42, 100),
            torch.rand(10, 42, 100),
            None,
            state=None
        )
        self.assertEqual(v.shape, (10, 100))
        self.assertEqual(s[0].shape, (10, 42, 4, 25))
        self.assertEqual(s[1].shape, (10, 42, 4, 25))

        v, s = att(
            torch.rand(10, 100),
            None,
            None,
            None,
            state=[torch.rand(10, 42, 4, 25), torch.rand(10, 42, 4, 25)]
        )
        self.assertEqual(v.shape, (10, 100))
        self.assertEqual(s[0].shape, (10, 42, 4, 25))
        self.assertEqual(s[1].shape, (10, 42, 4, 25))


if __name__ == "__main__":
    unittest.main()
