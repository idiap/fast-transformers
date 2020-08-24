#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import argparse
import unittest

from fast_transformers.builders import \
    TransformerEncoderBuilder, \
    RecurrentEncoderBuilder, \
    TransformerDecoderBuilder, \
    RecurrentDecoderBuilder


class TestTransformerEncoderBuilder(unittest.TestCase):
    def test_simple_build(self):
        transformer = TransformerEncoderBuilder().get()
        builder = TransformerEncoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = TransformerEncoderBuilder()
            builder.attention_type = "whatever"

    def test_builder_factory_methods(self):
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=4,
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
        builder.attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder.attention_type = "whatever"

    def test_decoder_build(self):
        transformer = TransformerDecoderBuilder().get()
        builder = TransformerDecoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.self_attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = TransformerDecoderBuilder()
            builder.self_attention_type = "whatever"
        with self.assertRaises(ValueError):
            builder = TransformerDecoderBuilder()
            builder.cross_attention_type = "whatever"

    def test_recurrent_decoder(self):
        transformer = RecurrentDecoderBuilder().get()
        builder = RecurrentDecoderBuilder()
        builder.n_layers = 1
        builder.n_heads = 4
        builder.self_attention_type = "linear"
        transformer = builder.get()

        with self.assertRaises(ValueError):
            builder = RecurrentDecoderBuilder()
            builder.self_attention_type = "whatever"
        with self.assertRaises(ValueError):
            builder = RecurrentDecoderBuilder()
            builder.cross_attention_type = "whatever"


if __name__ == "__main__":
    unittest.main()
