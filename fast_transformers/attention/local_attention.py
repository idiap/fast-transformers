#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement local context attention."""

from math import sqrt

import torch
from torch.nn import Module
from torch.nn import functional as F

from ..attention_registry import AttentionRegistry, Optional, Int, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..local_product import local_dot_product, local_weighted_average


class LocalAttention(Module):
    """TODO"""
    def __init__(self, local_context, softmax_temp=None, event_dispatcher=""):
        super(LocalAttention, self).__init__()
        self.local_context = local_context
        self.softmax_temp = softmax_temp
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """TODO"""
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
            attn_mask.additive_matrix,
            key_lengths.lengths,
            self.local_context
        )
        A = torch.softmax(softmax_temp * QK, dim=-1)

        V_new = local_weighted_average(A, values)

        return V_new.permute(0, 2, 1, 3).contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "local", LocalAttention,
    [
        ("local_context", Int),
        ("softmax_temp", Optional(Float)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
