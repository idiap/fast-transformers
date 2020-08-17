#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement the typical softmax attention as a recurrent cross attention
module to speed up autoregressive decoding."""

from math import sqrt

import torch
from torch.nn import Dropout, Module


class RecurrentCrossFullAttention(Module):
    """Implement autoregressive softmax cross attention as a recurrent
    module.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        dropout_rate: The dropout rate to apply to the attention (default: 0.1)
    """
    def __init__(self, softmax_temp=None, dropout_rate=0.1):
        super(RecurrentCrossFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(dropout_rate)

    def forward(self, query, keys, values, key_lengths, state=None):
        # Extract some shapes and compute the temperature
        N, H, E = query.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Extract the keys and values either from the arguments or the state
        if state is not None:
            keys, values = state

        # Compute the unnormalized attention and apply the key length mask
        QK = torch.einsum("nhe,nshe->nsh", query, keys)
        QK = QK + key_lengths.additive_matrix[:, :, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=1))
        V = torch.einsum("nsh,nshd->nhd", A, values)

        # Make sure that we return a contiguous value
        return V.contiguous(), [keys, values]
