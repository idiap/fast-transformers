#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import torch

from fast_transformers.events import EventDispatcher, QKVEvent, \
    AttentionEvent, IntermediateOutput
from fast_transformers.events.filters import layer_name_contains
from fast_transformers.builders import TransformerEncoderBuilder


class TestEvents(unittest.TestCase):
    def test_qkv(self):
        d = {}
        def store_qkv(event):
            d["q"] = event.queries
            d["k"] = event.keys
            d["v"] = event.values
        # default transformer is 4 layers 4 heads
        transformer = TransformerEncoderBuilder().get()
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(d), 0)

        EventDispatcher.get().listen(QKVEvent, store_qkv)
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(d), 3)
        d.clear()

        EventDispatcher.get().remove(store_qkv)
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(d), 0)
        d.clear()

        EventDispatcher.get().listen(
            QKVEvent & layer_name_contains(transformer, "layers.2.attention"),
            store_qkv
        )
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(d), 3)
        d.clear()

        EventDispatcher.get().listen(
            QKVEvent & layer_name_contains(transformer, "layers.22.attention"),
            store_qkv
        )
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(d), 0)
        d.clear()
        EventDispatcher.get().clear()

    def test_attention_matrix(self):
        A = []
        def store_attention(event):
            A.append(event.attention_matrix)
        # default transformer is 4 layers 4 heads
        transformer = TransformerEncoderBuilder().get()
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(A), 0)

        EventDispatcher.get().listen(AttentionEvent, store_attention)
        x = transformer(torch.rand(1, 100, 64*4))
        self.assertEqual(len(A), 4)

    def test_intermediate_output(self):
        intermediates = []
        def store_values(event):
            intermediates.append(event.x)

        transformer = TransformerEncoderBuilder().get()
        x = transformer(torch.rand(1, 100, 64*4))

        EventDispatcher.get().listen(IntermediateOutput, store_values)
        transformer(x)
        self.assertEqual(len(intermediates), 4)


if __name__ == "__main__":
    unittest.main()
