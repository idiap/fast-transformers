#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import torch

from fast_transformers.events import EventDispatcher, QKVEvent
from fast_transformers.events.filters import layer_name_contains
from fast_transformers.builders import TransformerEncoderBuilder


class TestEvents(unittest.TestCase):
    def test_qkv(self):
        d = {}
        def store_qkv(event):
            d["q"] = event.queries
            d["k"] = event.keys
            d["v"] = event.values
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


if __name__ == "__main__":
    unittest.main()
