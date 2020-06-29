#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the typical softmax attention as a recurrent module to speed up
autoregressive inference. See fast_transformers.attention.full_attention ."""

from math import sqrt

import torch
from torch.nn import Dropout, Module


class RecurrentFullAttention(Module):
    """Implement the full softmax attention as a recurrent module.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        dropout_rate: The dropout rate to apply to the attention (default: 0.1)
    """
    def __init__(self, softmax_temp=None, dropout_rate=0.1):
        super(RecurrentFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(dropout_rate)

    def forward(self, query, key, value, memory=None):
        # Extract some shapes and compute the temperature
        N, H, E = query.shape
        _, _, D = value.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Aggregate the list of keys and values
        if memory is not None:
            keys, values = memory
            keys = torch.cat([keys, key[:, :, None]], dim=2)
            values = torch.cat([values, value[:, :, None]], dim=2)
        else:
            keys = key[:, :, None]
            values = value[:, :, None]

        # Compute the unnormalized attention
        QK = torch.einsum("nhe,nhse->nhs", query, keys)

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhs,nhsd->nhd", A, values).contiguous()

        # Make sure that what we return is contiguous
        return V, [keys, values]
