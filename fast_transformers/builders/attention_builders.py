#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from collections import defaultdict

from .base import BaseBuilder
from ..attention_registry import \
    AttentionRegistry, \
    RecurrentAttentionRegistry, \
    RecurrentCrossAttentionRegistry


class BaseAttentionBuilder(BaseBuilder):
    def __init__(self, registry):
        self._registry = registry
        self._parameters = defaultdict(lambda: None)

    @property
    def available_attentions(self):
        """Return a list with the available attention implementations."""
        return self._registry.keys

    def validate_attention_type(self, attention_type):
        """Parse the attention type according to the rules used by `get()` and
        check if the requested attention is constructible."""
        return all(
            all(t in self._registry for t in a.split(","))
            for a in attention_type.split(":")
        )

    def __setattr__(self, key, value):
        # Make sure we have normal behaviour for the class members _registry
        # and _parameters
        if key in ["_registry", "_parameters"]:
            return object.__setattr__(self, key, value)

        # Assign everything else in the parameters dictionary
        if not self._registry.contains_parameter(key):
            raise AttributeError(("{!r} is not a valid attention "
                                  "parameter name").format(key))
        self._parameters[key] = self._registry.validate_parameter(key, value)

    def __getattr__(self, key):
        if key in self._parameters:
            return self._parameters[key]
        else:
            raise AttributeError()

    def __repr__(self):
        return (
            "{}.from_kwargs(\n".format(self.__class__.__name__) +
            "\n".join(["    {}={!r},".format(k, v)
                       for k, v in self._parameters.items()])[:-1] +
            "\n)"
        )

    def get(self, attention_type):
        """Construct the attention implementation object and return it.

        The passed in attention_type argument defines the attention to be
        created. It should be a string and in its simplest form it should
        be one of the available choices from `available_attentions`.

        However, to enable attention decoration, namely an attention
        implementation augmenting the functionality of another implementation,
        the attention type can be a colon separated list of compositions like
        the following examples:

            - 'att1' means instantiate att1
            - 'att2:att1' means instantiate att1 and decorate it with att2
            - 'att3:att1,att4' means instantiate att1 and att4 and decorate
              them with att3

        Arguments
        ---------
            attention_type: A string that contains one or more keys from
                            `available_attentions` separated with a colon to
                            denote the decoration pattern.
        """
        compositions = reversed(attention_type.split(":"))
        attentions = []
        for c in compositions:
            attentions = [
                self._construct_attention(t, attentions)
                for t in c.split(",")
            ]
        if len(attentions) > 1:
            raise ValueError(("Invalid attention_type argument "
                              "{!r}").format(attention_type))
        return attentions[0]

    def _construct_attention(self, attention_type, decorated=[]):
        """Construct an attention implementation object.

        Arguments
        ---------
            attention_type: A string that contains a single key from the
                            `available_attentions`
            decorated: A list of attention implementations to pass as arguments
                       to be decorated
        """
        if attention_type not in self._registry:
            raise ValueError(("Unknown attention type "
                              "{!r}").format(attention_type))

        attention, parameters = self._registry[attention_type]
        parameter_dictionary = {
            p: self._registry.validate_parameter(p, self._parameters[p])
            for p in parameters
        }

        return attention(*decorated, **parameter_dictionary)


class AttentionBuilder(BaseAttentionBuilder):
    """Build attention implementations for batch sequence processing or
    training."""
    def __init__(self):
        super(AttentionBuilder, self).__init__(AttentionRegistry)


class RecurrentAttentionBuilder(BaseAttentionBuilder):
    """Build attention implementations for autoregressive sequence
    processing."""
    def __init__(self):
        super(RecurrentAttentionBuilder, self).__init__(
            RecurrentAttentionRegistry
        )


class RecurrentCrossAttentionBuilder(BaseAttentionBuilder):
    """Build attention implementations for autoregressive cross attention
    computation."""
    def __init__(self):
        super(RecurrentCrossAttentionBuilder, self).__init__(
            RecurrentCrossAttentionRegistry
        )
