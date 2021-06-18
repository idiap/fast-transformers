#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement the attention proposed in 'An Attention Free Transformer'
(arxiv.org/abs/2105.14103)."""

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Int, \
    EventDispatcherInstance
from ..events import EventDispatcher


class AFTFullAttention(Module):
    """Implement the "full" attention proposed 'An Attention Free Transformer'.

    AFT attention uses only element wise operations to generate the new values
    given the queries, keys and values. The full AFT is computed as follows:

        v' = sigmoid(q) * (softmax(K + w_q, dim=1) * V).sum(dim=1)

    where q is a single query, K and V are the key and value matrices and w_q
    is a learnable vector for the given query position.

    Arguments
    ---------
        max_sequence_length: int, it defines the maximum acceptable sequence
                             length in order to allocate the learnable
                             parameters
        aft_parameterization: int, defines the dimensionality of the low rank
                              parameterization for the position bias
                              (default: 64)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, max_sequence_length=1024, aft_parameterization=64,
                 event_dispatcher=""):
        super().__init__()

        self.u = torch.nn.Parameter(
            torch.randn(max_sequence_length, aft_parameterization) * 0.01
        )
        self.v = torch.nn.Parameter(
            torch.randn(max_sequence_length, aft_parameterization) * 0.01
        )
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract some shapes
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        if E != D:
            raise ValueError(("AFT requires that queries, keys and values "
                              "have the same dimensionality"))

        # Make a gate out of Q and permute the dimensions of K and V
        Q = torch.sigmoid(queries)
        K = keys.permute(0, 2, 3, 1).contiguous()
        V = values.permute(0, 2, 3, 1).contiguous()

        # Mask the keys that are too long
        K = K + key_lengths.additive_matrix[:, None, None, :]

        # Compute the position bias and add it to the keys
        w = (self.u[:L].mm(self.v[:S].t())) + attn_mask.additive_matrix
        K = K[:, None, :, :, :] * w[None, :, None, None, :]

        # Compute the new values
        K = torch.softmax(K, dim=-1)
        V = Q * torch.einsum("nlhds,nhds->nlhd", K, V)

        return V


class AFTSimpleAttention(Module):
    """Implement the "simple" attention proposed 'An Attention Free Transformer'.

    AFT attention uses only element wise operations to generate the new values
    given the queries, keys and values. For the simple case that has no
    learnable parameters the new values are computed as follows:

        V' = sigmoid(Q) * (softmax(K, dim=1) * V).sum(dim=1)

    Arguments
    ---------
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, event_dispatcher=""):
        super().__init__()
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract some shapes
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        if E != D:
            raise ValueError(("AFT requires that queries, keys and values "
                              "have the same dimensionality"))

        # Make a gate out of Q
        Q = torch.sigmoid(queries)

        # Mask the keys that are too long
        K = keys + key_lengths.additive_matrix[:, :, None, None]

        # Autoregressive application
        if attn_mask.lower_triangular:
            # This is probably not the best strategy towards a cumulative
            # softmax because it might lead to underflow but it 'll have to do
            # until something better comes up
            M, _ = K.max(dim=1, keepdim=True)
            Kexp = torch.exp(K - M)
            Kexpsum = Kexp.cumsum(dim=1)
            K = Kexp / Kexpsum
            V = Q * (K * values).cumsum(dim=1)

        elif attn_mask.all_ones:
            K = torch.softmax(K, dim=1)
            V = Q * (K * values).sum(dim=1, keepdim=True)

        else:
            raise ValueError("You cannot use general attention masks with "
                             "AFTSimpleAttention because it would be "
                             "quadratic in time. Use AFTFullAttention "
                             "instead.")

        return V


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "aft-full", AFTFullAttention,
    [
        ("max_sequence_length", Optional(Int, 1024)),
        ("aft_parameterization", Optional(Int, 64)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
AttentionRegistry.register(
    "aft-simple", AFTSimpleAttention,
    [
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
