#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import torch

from fast_transformers.attention import AttentionLayer, FullAttention
from fast_transformers.masking import FullMask
from fast_transformers.transformers import TransformerDecoderLayer, \
    TransformerDecoder


class TestTransformerDecoder(unittest.TestCase):
    def test_full_attention_forward(self):
        d_model = 128
        n_heads = 4
        transformer = TransformerDecoder([
            TransformerDecoderLayer(
                AttentionLayer(FullAttention(), d_model, n_heads),  # self
                AttentionLayer(FullAttention(), d_model, n_heads),  # cross
                d_model
            )
            for i in range(6)
        ])
        x = torch.rand(10, 7, d_model)
        mem = torch.rand(10, 12, d_model)
        y = transformer(x, mem)
        self.assertEqual(y.shape, (10, 7, d_model))


if __name__ == "__main__":
    unittest.main()
