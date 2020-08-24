#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the causally masked linear attention as a recurrent model."""

import torch
from torch.nn import Module

from ....attention_registry import RecurrentAttentionRegistry, Optional, \
    Callable
from ..._utils import check_state


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class RecurrentLinearAttention(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, feature_map=None, eps=1e-6):
        super(RecurrentLinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)

        # Apply the feature map to the query and key
        Q = self.feature_map(query)
        K = self.feature_map(key)

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))
        else:
            Si, Zi = state

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K
            Si = Si + torch.einsum("nhd,nhm->nhdm", K, value)
        else:
            Zi += K
            Si += torch.einsum("nhd,nhm->nhdm", K, value)

        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        return V, [Si, Zi]


# Register the attention implementation so that it becomes available in our
# builders
RecurrentAttentionRegistry.register(
    "linear", RecurrentLinearAttention,
    [("feature_map", Optional(Callable))]
)
RecurrentAttentionRegistry.register(
    "causal-linear", RecurrentLinearAttention,
    [("feature_map", Optional(Callable))]
)
