#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


class Event(object):
    """The Event is the base class for all events that are dispatched from any
    transformer module.

    This class defines only the basic attributes of an event without any
    payload.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
    """
    def __init__(self, source):
        self.source = source


class AttentionEvent(Event):
    """An event containing an attention matrix.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        attention_matrix: torch.tensor of the multihead attention matrix
                          computed in the corresponding attention layer
    """
    def __init__(self, source, attention_matrix):
        super(AttentionEvent, self).__init__(source)
        self.attention_matrix = attention_matrix


class QKVEvent(Event):
    """An event containing the queries, keys and values projected in their
    multiple heads.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        queries: torch.tensor containing the queries in shape NLHE
        keys: torch.tensor containing the keys in shape NSHE
        values: torch.tensor containing the values in shape NSHD
    """
    def __init__(self, source, queries, keys, values):
        super(QKVEvent, self).__init__(source)
        self.queries = queries
        self.keys = keys
        self.values = values


class IntermediateOutput(Event):
    """Used by the TransformerEncoder and the TransformerDecoder to provide the
    intermediate outputs to interested callers.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        x: torch.tensor containing the intermediate features in shape NLD
    """
    def __init__(self, source, x):
        super().__init__(source)
        self.x = x
