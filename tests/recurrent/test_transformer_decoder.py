#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


import unittest

import torch

from fast_transformers.attention import AttentionLayer, FullAttention, \
    LinearAttention, CausalLinearAttention
from fast_transformers.masking import TriangularCausalMask, FullMask, \
    LengthMask
from fast_transformers.recurrent.attention import \
    RecurrentAttentionLayer, RecurrentCrossAttentionLayer, \
    RecurrentFullAttention, RecurrentCrossFullAttention, \
    RecurrentLinearAttention, RecurrentCrossLinearAttention
from fast_transformers.recurrent.transformers import \
    RecurrentTransformerDecoderLayer, RecurrentTransformerDecoder
from fast_transformers.transformers import TransformerDecoderLayer, \
    TransformerDecoder


class TestRecurrentTransformerDecoder(unittest.TestCase):
    def test_compare_with_batch(self):
        tests = [
            ("full", FullAttention, FullAttention, 
             RecurrentFullAttention, RecurrentCrossFullAttention),
            ("linear", CausalLinearAttention, LinearAttention,
             RecurrentLinearAttention, RecurrentCrossLinearAttention)
        ]

        N = 10
        L = 42
        S = 100
        D = 1024
        x = torch.rand(N, L, D)
        m = torch.rand(N, S, D)

        for name, a1, a2, a3, a4 in tests:
            dec = TransformerDecoder([
                TransformerDecoderLayer(
                    AttentionLayer(a1(), D, 4),
                    AttentionLayer(a2(), D, 4),
                    D
                )
                for i in range(4)
            ])
            rdec = RecurrentTransformerDecoder([
                RecurrentTransformerDecoderLayer(
                    RecurrentAttentionLayer(a3(), D, 4),
                    RecurrentCrossAttentionLayer(a4(), D, 4),
                    D
                )
                for i in range(4)
            ])
            dec.eval()
            rdec.eval()
            rdec.load_state_dict(dec.state_dict())

            x_mask = TriangularCausalMask(L)
            x_length = LengthMask(torch.full((N,), L, dtype=torch.int64))
            m_mask = FullMask(L, S)
            m_length = LengthMask(torch.full((N,), S, dtype=torch.int64))

            y1 = dec(x, m, x_mask=x_mask, x_length_mask=x_length,
                     memory_mask=m_mask, memory_length_mask=m_length)
            state = None
            y2 = []
            for i in range(L):
                y2i, state = rdec(x[:, i], m, memory_length_mask=m_length,
                                  state=state)
                y2.append(y2i)
            y2 = torch.stack(y2, dim=1)

            self.assertLess(torch.abs(y1-y2).max(), 1e-5)

    def test_mask_creation(self):
        N = 10
        L = 42
        S = 100
        D = 1024
        x = torch.rand(N, D)
        m = torch.rand(N, S, D)

        rdec = RecurrentTransformerDecoder([
            RecurrentTransformerDecoderLayer(
                RecurrentAttentionLayer(RecurrentFullAttention(), D, 4),
                RecurrentCrossAttentionLayer(
                    RecurrentCrossFullAttention(), D, 4
                ),
                D
            )
            for i in range(4)
        ])
        rdec(x, m)


if __name__ == "__main__":
    unittest.main()
