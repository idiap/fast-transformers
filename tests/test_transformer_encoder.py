#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

from fast_transformers.attention import AttentionLayer, FullAttention, \
    ClusteredAttention, ImprovedClusteredAttention, ReformerAttention
from fast_transformers.masking import FullMask
from fast_transformers.transformers import TransformerEncoderLayer, TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
    def test_full_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(FullAttention(), d_model, n_heads),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        x = transformer(torch.rand(10, 7, d_model))
        self.assertEqual(x.shape, (10, 7, d_model))

    def test_clustered_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(
                    ClusteredAttention(
                        clusters = 10
                    ),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        x = transformer(torch.rand(100, 20, d_model))
        self.assertEqual(x.shape, (100, 20, d_model))

    def test_improved_clustered_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(
                    ImprovedClusteredAttention(
                        clusters=10,
                        topk=5
                    ),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        x = torch.rand(100, 20, d_model)
        y = transformer(x)
        self.assertEqual(y.shape, (100, 20, d_model))

    def test_improved_clustered_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(
                    ReformerAttention(
                        chunk_size=32,
                        rounds=4,
                        bits=8,
                        masked=False,
                    ),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        x = torch.rand(12, 128, d_model)
        y = transformer(x)
        self.assertEqual(y.shape, (12, 128, d_model))


if __name__ == "__main__":
    unittest.main()
