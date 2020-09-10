#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Spec instances allow to describe and check the type and value of
parameters."""

from ..events import EventDispatcher


class Spec(object):
    """Describe and validate a parameter type.

    Arguments
    ---------
        predicate: A callable that checks if the value is acceptable and
                   returns its canonical value or raises ValueError.
        name: A name to create a human readable description of the Spec
    """
    def __init__(self, predicate, name="CustomSpec"):
        self._predicate = predicate
        self._name = name

    def __repr__(self):
        return self._name

    def check(self, x):
        try:
            self._predicate(x)
            return True
        except ValueError:
            return False

    def get(self, x):
        return self._predicate(x)

    def __eq__(self, y):
        return self is y


class Choice(Spec):
    """A parameter type for a set of options.

    Arguments
    ---------
        choices: A set or list of possible values for this parameter
    """
    def __init__(self, choices):
        self._choices = choices

    def get(self, x):
        if x in self._choices:
            return x
        raise ValueError("{!r} is not in {!r}".format(x, self._choices))

    def __repr__(self):
        return "Choice({!r})".format(self._choices)

    def __eq__(self, x):
        if isinstance(x, Choice):
            return self._choices == x._choices
        return False


class _Callable(Spec):
    def __init__(self):
        super(_Callable, self).__init__(None, "Callable")

    def get(self, x):
        if callable(x):
            return x
        raise ValueError("{!r} is not a callable".format(x))


class _EventDispatcherInstance(Spec):
    def __init__(self):
        super(_EventDispatcherInstance, self).__init__(
            _EventDispatcherInstance._get_event_dispatcher,
            "EventDispatcherInstance"
        )

    @staticmethod
    def _get_event_dispatcher(x):
        if isinstance(x, str):
            return x
        if isinstance(x, EventDispatcher):
            return x
        raise ValueError("{!r} is not an event dispatcher".format(x))


class Optional(Spec):
    """Represent an optional parameter that can either have a value or it can
    be None.

    Arguments
    ---------
        spec: The spec for the value if it is not None
        default: The returned value in case it is None
    """
    def __init__(self, spec, default=None):
        self._other_spec = spec
        self._default = default

    def __repr__(self):
        return "Optional[{!r}, {!r}]".format(self._other_spec, self._default)

    def get(self, x):
        if x is None:
            return self._default
        return self._other_spec.get(x)

    def __eq__(self, x):
        if isinstance(x, Optional):
            return (
                self._other_spec == x._other_spec and
                self._default == x._default
            )
        return False


Int = Spec(int, "Int")
Float = Spec(float, "Float")
Bool = Spec(bool, "Bool")
Callable = _Callable()
EventDispatcherInstance = _EventDispatcherInstance()
