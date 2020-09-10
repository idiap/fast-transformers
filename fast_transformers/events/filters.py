#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Define composable functions to filter events."""

import weakref

from .event import Event


class EventFilter(object):
    """EventFilter instances are predicates (ie functions that return True or
    False) to be used with an event dispatcher for filtering event
    instances.

    The main benefit from using raw functions is that an EventFilter composes
    very easily using operators such as &, |, ~.

    Example
    --------

        event_filter = AttentionEvent | layer_name_contains("layers.1")
        event_filter = from_layer(transformer.layers[2].attention)
        event_filter = (
            AttentionEvent &
            lambda ev: torch.isnan(ev.attention_matrix).any()
        )
    """
    def __call__(self, event):
        raise NotImplementedError()

    def _to_event_filter(self, other):
        if isinstance(other, EventFilter):
            return other
        if isinstance(other, type) and issubclass(other, Event):
            return event_class(other)
        if callable(other):
            return CallableEventFilter(other)

        return NotImplemented

    def __and__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) and other(ev))

    def __rand__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) and self(ev))

    def __or__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) or other(ev))

    def __ror__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) or self(ev))

    def __invert__(self):
        return CallableEventFilter(lambda ev: not self(ev))


class CallableEventFilter(EventFilter):
    """Wrap a function with an EventFilter object."""
    def __init__(self, event_filter):
        self._event_filter = event_filter

    def __call__(self, event):
        return self._event_filter(event)


class LayerNameEventFilter(EventFilter):
    """A LayerNameEventFilter allows to filter events based on a human readable
    name of the layer that emitted them.

    Note that LayerNameEventFilter keeps a weak reference to all modules which
    means that it cannot be used to prevent modules from being garbage
    collected.

    Arguments
    ---------
        root: torch.nn.Module instance that represents the root container
        name_filter: callable, that returns true if the name 
    """
    def __init__(self, root, name_filter):
        self._names = {
            weakref.ref(m): n
            for n, m in root.named_modules()
        }
        self._name_filter = name_filter

    def __call__(self, event):
        name = self._names.get(weakref.ref(event.source), None)
        if name is None:
            return False
        return self._name_filter(name)


def event_class(klass):
    """Select events that are instances of `klass`.

    Arguments
    ---------
        klass: A class to check the event instance against

    Returns
    -------
        An instance of EventFilter
    """
    return CallableEventFilter(lambda ev: isinstance(ev, klass))


def from_layer(layer):
    """Select events that are dispatched from the `layer`.

    Arguments
    ---------
        layer: An instance of torch.nn.Module to check against the event source

    Returns
    -------
        An instance of EventFilter
    """
    return CallableEventFilter(lambda ev: ev.source is layer)


def layer_name_contains(root, name):
    """Select events that contain `name` in their human readable name.

    We use root.named_modules() to get human readable names for the layers.
    """
    return LayerNameEventFilter(root, lambda n: name in n)
