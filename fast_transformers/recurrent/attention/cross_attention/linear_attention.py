#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement unmasked linear attention as a recurrent cross attention module to
speed up autoregressive decoding."""

import torch
from torch.nn import Module

from ....attention_registry import RecurrentCrossAttentionRegistry, Optional, \
    Callable


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class RecurrentCrossLinearAttention(Module):
    """Implement autoregressive linear cross attention as a recurrent
    module.

    See fast_transformers.attention.linear_attention.LinearAttention .

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, feature_map=None, eps=1e-6):
        super(RecurrentCrossLinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, query, keys, values, key_lengths, state=None):
        # Compute the feature representation of the query
        Q = self.feature_map(query)

        # If the state is not given compute the key-value matrix and the
        # normalizers, namely compute whatever is needed in order to attend to
        # keys and values with a given query.
        if state is None:
            K = self.feature_map(keys)
            K = K * key_lengths.float_matrix[:, :, None, None]
            S = torch.einsum("nshd,nshm->nhmd", K, values)
            Z = K.sum(dim=1)
        else:
            S, Z = state

        # Given S and Z now we can efficiently compute the new value
        QZ = 1/(torch.einsum("nhd,nhd->nh", Q, Z)+self.eps)
        V = torch.einsum("nhd,nhmd,nh->nhm", Q, S, QZ)

        return V.contiguous(), [S, Z]


# Register the attention implementation so that it becomes available in our
# builders
RecurrentCrossAttentionRegistry.register(
    "linear", RecurrentCrossLinearAttention,
    [("feature_map", Optional(Callable))]
)
