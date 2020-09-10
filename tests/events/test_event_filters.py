#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import torch

from fast_transformers.events import Event
from fast_transformers.events.filters import event_class, from_layer, \
    layer_name_contains


class MockEvent(Event):
    def __init__(self, source, payload):
        super(MockEvent, self).__init__(source)
        self.payload = payload


class TestEventFilters(unittest.TestCase):
    def test_simple_filters(self):
        mock_event = event_class(MockEvent)
        self.assertTrue(mock_event(MockEvent(None, None)))
        self.assertFalse(mock_event(Event(None)))

        source = object()
        fl = from_layer(source)
        self.assertTrue(fl(Event(source)))
        self.assertFalse(fl(Event(None)))
        self.assertFalse(fl(Event(object())))

        net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )
        lnc = layer_name_contains(net, "2.1")
        self.assertTrue(lnc(Event(net[2][1])))

    def test_filter_composition(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

        event_filter = MockEvent & from_layer(net[2][1])
        self.assertFalse(event_filter(Event(net[2][1])))
        self.assertFalse(event_filter(MockEvent(net[2], None)))
        self.assertTrue(event_filter(MockEvent(net[2][1], None)))

        # should raise error because ev.payload is accessed before event is
        # made sure to be a MockEvent
        event_filter = (lambda ev: ev.payload == 0) & event_class(MockEvent)
        with self.assertRaises(AttributeError):
            event_filter(Event(None))
        event_filter = event_class(MockEvent) & (lambda ev: ev.payload == 0)
        self.assertFalse(event_filter(Event(None)))
        self.assertFalse(event_filter(MockEvent(None, 1)))
        self.assertTrue(event_filter(MockEvent(None, 0)))


if __name__ == "__main__":
    unittest.main()
