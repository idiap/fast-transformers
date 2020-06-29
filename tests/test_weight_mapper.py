#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch
import torch.nn as nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.weight_mapper import PytorchMapper, \
    HugginfaceBertEncoderMapper, LongformerMapper

try:
    from transformers import BertConfig, BertModel
except ImportError:
    BertConfig = BertModel = None

try:
    from longformer.longformer import LongformerConfig, Longformer
except ImportError:
    LongformerConfig = Longformer = None


class TestWeightMapper(unittest.TestCase):
    def test_mapping(self):
        t1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                128, 4, dim_feedforward=256
            ),
            4
        )
        t2 = TransformerEncoderBuilder.from_kwargs(
            n_layers=4,
            n_heads=4,
            query_dimensions=128//4,
            value_dimensions=128//4,
            feed_forward_dimensions=256,
            attention_type="full",
            final_normalization=False
        ).get()
        t1.eval()
        t2.eval()

        with self.assertRaises(RuntimeError):
            t2.load_state_dict(t1.state_dict())

        t2.load_state_dict(PytorchMapper().map(t1.state_dict()))
        x = torch.rand(3, 10, 128)
        o1 = t2(x)
        o2 = t1(x.permute(1, 0, 2)).permute(1, 0, 2)
        self.assertLess(torch.abs(o1 - o2).max().item(), 1e-5)

    @unittest.skipUnless(BertConfig, "Hugginface is not installed")
    def test_huggin_bert(self):
        bert = BertModel(BertConfig())
        encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=12,
            n_heads=12,
            query_dimensions=64,
            value_dimensions=64,
            feed_forward_dimensions=3072,
            attention_type="full",
            final_normalization=False,
            activation="gelu"
        ).get()
        bert.eval()
        encoder.eval()

        # Before the weight copy they should be different
        x = torch.rand(3, 10, 768)
        o1 = bert.encoder(x, head_mask=[None]*12)[0]
        o2 = encoder(x)
        self.assertGreater(torch.abs(o1-o2).max().item(), 1)

        # And after the copy they should be exactly the same
        encoder.load_state_dict(
            HugginfaceBertEncoderMapper().map(bert.encoder.state_dict())
        )
        o1 = bert.encoder(x, head_mask=[None]*12)[0]
        o2 = encoder(x)
        self.assertLess(torch.abs(o1-o2).max().item(), 1e-4)

    @unittest.skipUnless(Longformer, "Longformer is not installed")
    def test_longformer(self):
        config = LongformerConfig()
        config.attention_mode = "n2"
        config.attention_window = [256]*12
        config.attention_dilation = [1]*12
        longformer = Longformer(config)
        encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=12,
            n_heads=12,
            query_dimensions=64,
            value_dimensions=64,
            feed_forward_dimensions=3072,
            attention_type="full",
            final_normalization=False,
            activation="gelu"
        ).get()
        longformer.eval()
        encoder.eval()

        # Before the weight copy they should be different
        x = torch.rand(3, 10, 768)
        o1 = longformer.encoder(x, head_mask=[None]*12)[0]
        o2 = encoder(x)
        self.assertGreater(torch.abs(o1-o2).max().item(), 1)

        # And after the copy they should be exactly the same
        encoder.load_state_dict(
            LongformerMapper().map(longformer.encoder.state_dict())
        )
        o1 = longformer.encoder(x, head_mask=[None]*12)[0]
        o2 = encoder(x)
        self.assertLess(torch.abs(o1-o2).max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
