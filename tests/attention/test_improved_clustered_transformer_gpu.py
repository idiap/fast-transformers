#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

from fast_transformers.attention import AttentionLayer, \
    ImprovedClusteredAttention, FullAttention
from fast_transformers.masking import FullMask
from fast_transformers.masking import LengthMask
from fast_transformers.transformers import TransformerEncoderLayer, \
    TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
    def test_full_attention_forward(self):
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
        transformer = transformer.to("cuda")
        x = torch.rand(100, 20, d_model).to("cuda")
        y = transformer(x)
        self.assertEqual(y.shape, (100, 20, d_model))

    def test_topk_equals_length_attention(self):
        d_model = 32
        n_heads = 4
        improved_transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(
                    ImprovedClusteredAttention(
                        clusters=10,
                        topk=20
                    ),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        full_transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(FullAttention(), d_model, n_heads),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        full_transformer = full_transformer.to("cuda")
        improved_transformer = improved_transformer.to("cuda")
        improved_transformer.load_state_dict(full_transformer.state_dict())
        improved_transformer.eval()
        full_transformer.eval()
        x = torch.rand(100, 20, d_model).to("cuda")
        y_full = improved_transformer(x)
        y_improved = full_transformer(x)
        self.assertLess(
            torch.max(torch.abs(y_improved - y_full)),
            1e-4
        )

    def test_topk_equals_length_attention_masked(self):
        d_model = 32
        n_heads = 4
        improved_transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(
                    ImprovedClusteredAttention(
                        clusters=10,
                        topk=20
                    ),
                    d_model,
                    n_heads
                ),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        full_transformer = TransformerEncoder([
            TransformerEncoderLayer(
                AttentionLayer(FullAttention(), d_model, n_heads),
                d_model,
                n_heads
            )
            for i in range(6)
        ])
        full_transformer = full_transformer.to("cuda")
        improved_transformer = improved_transformer.to("cuda")
        improved_transformer.load_state_dict(full_transformer.state_dict())
        improved_transformer.eval()
        full_transformer.eval()
        x = torch.rand(100, 20, d_model).to("cuda")
        lengths = x.new_full((100,), 20, dtype=torch.int64)
        lengths[1] = 5
        lengths[10] = 10
        length_mask = LengthMask(
            lengths=lengths,
            max_len=20
        )
        y_full = improved_transformer(x, length_mask=length_mask)
        y_improved = full_transformer(x, length_mask=length_mask)
        self.assertLess(
            torch.max(torch.abs(y_improved[1,:5] - y_full[1,:5])),
            1e-4
        )
        self.assertLess(
            torch.max(torch.abs(y_improved[10,:10] - y_full[10,:10])),
            1e-4
        )

if __name__ == "__main__":
    unittest.main()
