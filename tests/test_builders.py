#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import argparse
import unittest

import torch

from fast_transformers.builders import \
    TransformerEncoderBuilder, \
    RecurrentEncoderBuilder, \
    TransformerDecoderBuilder, \
    RecurrentDecoderBuilder
from fast_transformers.attention_registry import AttentionRegistry, Int


class TestAttention:
    def __init__(self, n_heads, query_dimensions):
        self.n_heads = n_heads
        self.query_dimensions = query_dimensions

AttentionRegistry.register(
    "test-attention", TestAttention,
    [
        ("n_heads", Int),
        ("query_dimensions", Int)
    ]
)


class TestBuilders(unittest.TestCase):
    def test_simple_build(self):
        transformer = TransformerEncoderBuilder().get()
        builder = TransformerEncoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.query_dimensions = 32
        builder.attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = TransformerEncoderBuilder()
            builder.attention_type = "whatever"

    def test_builder_factory_methods(self):
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=4,
            query_dimensions=32,
            attention_type="linear"
        )

        with self.assertRaises(ValueError):
            TransformerEncoderBuilder.from_kwargs(
                foobar=1
            )
        TransformerEncoderBuilder.from_kwargs(
            foobar=1,
            strict=False
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_layers", type=int)
        parser.add_argument("--n_heads", type=int)
        args = parser.parse_args(["--n_heads", "42"])
        builder = TransformerEncoderBuilder.from_namespace(args)
        self.assertEqual(builder.n_heads, 42)
        self.assertTrue(builder.n_layers is None)

    def test_recurrent_build(self):
        transformer = RecurrentEncoderBuilder().get()
        builder = RecurrentEncoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.query_dimensions = 32
        builder.attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder.attention_type = "whatever"

    def test_decoder_build(self):
        transformer = TransformerDecoderBuilder().get()
        builder = TransformerDecoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.query_dimensions = 32
        builder.self_attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = TransformerDecoderBuilder()
            builder.self_attention_type = "whatever"
        with self.assertRaises(ValueError):
            builder = TransformerDecoderBuilder()
            builder.cross_attention_type = "whatever"

        builder.cross_n_heads = 7
        builder.cross_value_dimensions = 32
        transformer = builder.get()
        x = torch.rand(1, 20, 4*64)
        m = torch.rand(1, 13, 7*32)
        y = transformer(x, m)

        t = TransformerDecoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=4,
            query_dimensions=32,
            cross_n_heads=7,
            cross_value_dimensions=32,
            cross_query_dimensions=32
        ).get()

    def test_recurrent_decoder(self):
        transformer = RecurrentDecoderBuilder().get()
        builder = RecurrentDecoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.query_dimensions = 32
        builder.self_attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = RecurrentDecoderBuilder()
            builder.self_attention_type = "whatever"
        with self.assertRaises(ValueError):
            builder = RecurrentDecoderBuilder()
            builder.cross_attention_type = "whatever"

        builder.cross_n_heads = 7
        builder.cross_value_dimensions = 32
        transformer = builder.get()
        x = torch.rand(1, 4*64)
        m = torch.rand(1, 13, 7*32)
        y, s = transformer(x, m)
        y, s = transformer(x, m, state=s)

    def test_attention_parameter(self):
        builder = TransformerEncoderBuilder()

        builder.n_layers = 3
        builder.n_heads = 4
        builder.feed_forward_dimensions = 512
        builder.query_dimensions = 32
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.activation = "relu"
        builder.final_normalization = True

        # Full attention parameters
        builder.softmax_temp = 1.0
        builder.attention_dropout = 0.1

        # Linear attention parameters
        builder.feature_map = lambda x: (x > 0).float() * x

        # Clustered attention parameters
        builder.clusters = 100
        builder.iterations = 10
        builder.bits = 32
        builder.hash_bias = True

        # Exact topk attention parameters
        builder.topk = 32

        # Conditional attention parameters
        builder.length_limit = 512

        # Reformer attention parameters
        builder.chunk_size = 32
        builder.rounds = 1

        # Add here old parameters to avoid regressions
        invalid = [
            "dropout_rate"
        ]
        for name in invalid:
            with self.assertRaises(AttributeError):
                setattr(builder, name, None)

    def test_attention_composition(self):
        transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type="conditional-full:improved-clustered",
            attention_dropout=0.1,
            softmax_temp=0.125,
            clusters=256,
            bits=32,
            topk=32,
            length_limit=512
        ).get()

        with self.assertRaises(TypeError):
            transformer = TransformerEncoderBuilder.from_kwargs(
                attention_type="conditional-full",
                attention_dropout=0.1,
                softmax_temp=0.125,
                length_limit=512
            ).get()

    def test_transformer_parameters_to_attention(self):
        with self.assertRaises(ValueError):
            transformer = TransformerEncoderBuilder.from_kwargs(
                attention_type="test-attention"
            ).get()

        transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type="test-attention",
            n_heads=8,
            query_dimensions=64
        ).get()


if __name__ == "__main__":
    unittest.main()
