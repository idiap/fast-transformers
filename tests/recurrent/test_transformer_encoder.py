#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

from fast_transformers.recurrent.attention import RecurrentAttentionLayer, \
    RecurrentFullAttention, RecurrentLinearAttention
from fast_transformers.recurrent.transformers import \
    RecurrentTransformerEncoderLayer, RecurrentTransformerEncoder


class TestRecurrentTransformerEncoder(unittest.TestCase):
    def test_full_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = RecurrentTransformerEncoder([
            RecurrentTransformerEncoderLayer(
                RecurrentAttentionLayer(
                    RecurrentFullAttention(),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])

        xs = []
        memory = None
        for i in range(7):
            x, memory = transformer(torch.rand(10, d_model), state=memory)
            xs.append(x)
        for i in range(7):
            self.assertEqual(xs[i].shape, (10, d_model))
        self.assertEqual(len(memory), 6)
        for i in range(6):
            self.assertEqual(len(memory[i]), 2)
            self.assertEqual(memory[i][0].shape, (10, n_heads, 7, 32))
            self.assertEqual(memory[i][1].shape, (10, n_heads, 7, 32))

    def test_linear_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = RecurrentTransformerEncoder([
            RecurrentTransformerEncoderLayer(
                RecurrentAttentionLayer(
                    RecurrentLinearAttention(),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])

        xs = []
        memory = None
        for i in range(7):
            x, memory = transformer(torch.rand(10, d_model), state=memory)
            xs.append(x)
        for i in range(7):
            self.assertEqual(xs[i].shape, (10, d_model))
        self.assertEqual(len(memory), 6)
        for i in range(6):
            self.assertEqual(len(memory[i]), 2)
            self.assertEqual(memory[i][0].shape, (10, n_heads, 32, 32))
            self.assertEqual(memory[i][1].shape, (10, n_heads, 32))


if __name__ == "__main__":
    unittest.main()
