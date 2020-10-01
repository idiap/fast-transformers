#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement local context attention."""

from math import sqrt

import torch
from torch.nn import Module, Dropout
from torch.nn import functional as F

from ..attention_registry import AttentionRegistry, Optional, Int, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..local_product import local_dot_product, local_weighted_average


class LocalAttention(Module):
    """Implement fast local attention where a query can only attend to
    neighboring keys.

    In this attention module the query Q_i can only attend to a key K_j if
    |i-j| < local_context/2.

    Arguments
    ---------
        local_context: The neighborhood to consider for local attention.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, local_context, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(LocalAttention, self).__init__()
        self.local_context = local_context
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the local attention.

        The attn_mask can be anything but the only values that will be
        considered will be the ones in the neighborhood of each query.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        context = self.local_context
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Permute the dimensions to NHLE instead of NLHE
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()

        QK = local_dot_product(
            queries,
            keys,
            attn_mask.additive_matrix_finite,
            key_lengths.lengths,
            self.local_context
        )
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))

        V_new = local_weighted_average(A, values)

        return V_new.permute(0, 2, 1, 3).contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "local", LocalAttention,
    [
        ("local_context", Int),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
