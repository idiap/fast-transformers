#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the oracle top-k attention. The top-k keys are exact ones.
MultiHeadAttention module. Note that this module is to be used in conjuction
with the AttentionLayer in order to work."""

from math import sqrt

import torch
from torch.nn import Dropout, Module

from ..attention_registry import AttentionRegistry, Optional, Int, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher


class ExactTopKAttention(Module):
    """Implement the oracle top-k softmax attention.

    Arguments
    ---------
        top-k: The top k keys to attend to  (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, topk=32, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(ExactTopKAttention, self).__init__()
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        topk = min(self.topk, S)
        
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        topk_values, topk_idx = torch.topk(QK, topk, sorted=False, dim=-1)
        mask = QK.new_ones(QK.shape) *  float("-inf") 
        mask[
            torch.arange(N, device=QK.device).view(N, 1, 1, 1),
            torch.arange(H, device=QK.device).view(1, H, 1, 1),
            torch.arange(L, device=QK.device).view(1, 1, L, 1),
            topk_idx,
        ] = 0.

        QK = QK + mask 

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "exact-topk", ExactTopKAttention,
    [
        ("topk", Optional(Int, 32)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
