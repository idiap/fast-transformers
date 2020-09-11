#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from collections import OrderedDict

from .event import Event
from .filters import event_class


class EventDispatcher(object):
    """An EventDispatcher is a simple way to implement an observer pattern for
    loose coupling of components. In our case it is used so that the internals
    of large neural networks can communicate with the outside world in an
    agnostic and efficient way.

    Example usage
    -------------

        from fast_transformers.events import EventDispatcher, AttentionEvent
        from fast_transformers.events.filters import \
            layer_name_contains

        def attention_event_handler(event):
            print(event.attention_matrix)

        ed = EventDispatcher()
        ed.listen(AttentionEvent, attention_event_handler)
        ed.listen(
            AttentionEvent & layer_name_contains("layers.12"),
            attention_event_handler
        )
    """
    _dispatchers = {}

    def __init__(self):
        self._listeners = OrderedDict()

    def listen(self, event_filter, event_handler):
        """Add an event handler for the events that pass the event filter.

        Arguments
        ---------
            event_filter: callable or Event class to define for which events
                          this handler will be called
            event_handler: callable that accepts an instance of Event
        """
        if isinstance(event_filter, type) and issubclass(event_filter, Event):
            event_filter = event_class(event_filter)

        self._listeners[event_handler] = event_filter

    def remove(self, event_handler):
        """Remove the event_handler from the listeners so that no more events
        are dispatched to this handler."""
        self._listeners.pop(event_handler, None)

    def clear(self):
        """Remove all listeners from the event dispatcher."""
        self._listeners.clear()

    def dispatch(self, event):
        """Dispatch an event to the listeners.

        Arguments
        ---------
            event: Event instance
        """
        for event_handler, event_filter in self._listeners.items():
            if event_filter(event):
                event_handler(event)

    @classmethod
    def get(cls, key=""):
        """Factory method for creating global event dispatchers for loosely
        coupling parts of a larger codebase.

        Since global objects are a complete antipattern, we suggest that this
        is only used to set a default value for an event dispatcher passed as
        an argument.

        Argument
        --------
            key: A key to uniquely identify a dispatcher or an instance of a
                 dispatcher to be returned as is
        """
        if isinstance(key, cls):
            return key
        if key not in cls._dispatchers:
            cls._dispatchers[key] = cls()
        return cls._dispatchers[key]
