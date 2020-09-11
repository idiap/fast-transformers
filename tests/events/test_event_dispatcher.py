#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from fast_transformers.events import Event, EventDispatcher


class MockEvent(Event):
    def __init__(self, source, payload):
        super(MockEvent, self).__init__(source)
        self.payload = payload


class TestEventDispatcher(unittest.TestCase):
    def test_simple_listen_dispatch(self):
        d = {"x": 0}
        def listener1(event):
            d["x"] += 1

        def listener2(event):
            d["x"] += 1

        ed = EventDispatcher()
        ed.listen(Event, listener1)
        ed.listen(Event, listener2)
        ed.dispatch(Event(None))
        self.assertEqual(d["x"], 2)
        ed.remove(listener1)
        ed.dispatch(Event(None))
        self.assertEqual(d["x"], 3)
        ed.remove(listener2)

        def set_payload(event):
            d.update(event.payload)
        ed.listen(MockEvent, set_payload)
        ed.dispatch(Event(None))
        self.assertTrue("y" not in d)
        ed.dispatch(MockEvent(None, {"y": 0}))
        self.assertEqual(d["y"], 0)
        self.assertEqual(d["x"], 3)

    def test_factory_method(self):
        ed1 = EventDispatcher.get()
        ed2 = EventDispatcher.get()
        self.assertTrue(ed1 is ed2)
        ed1 = EventDispatcher.get("foo")
        ed2 = EventDispatcher.get("bar")
        self.assertTrue(ed1 is not ed2)
        ed1 = EventDispatcher.get("foo")
        ed2 = EventDispatcher.get("foo")
        self.assertTrue(ed1 is ed2)
        ed1 = EventDispatcher()
        ed2 = EventDispatcher.get(ed1)
        self.assertTrue(ed1 is ed2)



if __name__ == "__main__":
    unittest.main()
