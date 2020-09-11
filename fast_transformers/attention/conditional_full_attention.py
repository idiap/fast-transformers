#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement a self attention that delegates to full attention or another
attention depending on the input sequence length."""

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Int, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher
from .full_attention import FullAttention


class ConditionalFullAttention(Module):
    """"Delegate to full attention if the input sequence is short.

    Arguments
    ---------
        other_attention: Use the passed attention module if the sequence is
                         longer than 'length_limit'.
        length_limit: An integer denoting the maximum sequence length to
                      consider.
        softmax_temp: See fast_transformers.attention.full_attention.
        attention_dropout: See fast_transformers.attention.full_attention.
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, other_attention, length_limit=512, softmax_temp=None,
                 attention_dropout=0.1, event_dispatcher=""):
        super(ConditionalFullAttention, self).__init__()
        self.full_attention = FullAttention(softmax_temp, attention_dropout)
        self.other_attention = other_attention
        self.length_limit = length_limit
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract some shapes to compare with the length limit
        L = queries.shape[1]
        S = values.shape[1]

        if L > self.length_limit or S > self.length_limit:
            return self.other_attention(queries, keys, values, attn_mask,
                                        query_lengths, key_lengths)
        else:
            return self.full_attention(queries, keys, values, attn_mask,
                                       query_lengths, key_lengths)


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "conditional-full", ConditionalFullAttention,
    [
        ("length_limit", Optional(Int, 512)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
