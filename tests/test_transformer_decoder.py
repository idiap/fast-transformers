#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import os
import time
import unittest

import torch

from fast_transformers.attention import AttentionLayer, FullAttention
from fast_transformers.builders import RecurrentDecoderBuilder
from fast_transformers.masking import FullMask, LengthMask
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

    @unittest.skipUnless(os.getenv("BENCHMARK_TESTS"), "no benchmarks")
    def test_decoder_inference_benchmark(self):
        builder = RecurrentDecoderBuilder.from_kwargs(
            n_layers=4,
            n_heads=8,
            query_dimensions=64,
            value_dimensions=64
        )
        t1 = builder.get()
        builder.self_attention_type = "linear"
        builder.cross_attention_type = "linear"
        t2 = builder.get()

        B = 128
        L = 100
        S = 100
        D = 512
        memory = torch.rand(B, S, D)
        memory_lengths = LengthMask(torch.full((B,), S, dtype=torch.int64))

        x = torch.rand(B, D)
        state = None
        start = time.time()
        with torch.no_grad():
            for i in range(L):
                x, state = t1(x, memory, memory_lengths, state=state)
        end = time.time()
        print("Softmax attention took", round(end-start, 2), "s")

        x = torch.rand(B, D)
        state = None
        start = time.time()
        with torch.no_grad():
            for i in range(L):
                x, state = t2(x, memory, memory_lengths, state=state)
        end = time.time()
        print("Linear attention took", round(end-start, 2), "s")


if __name__ == "__main__":
    unittest.main()
