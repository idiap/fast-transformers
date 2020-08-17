#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Autoregressive implementations for cross attention as a recurrent module.

The attention implementations in this module expect one input for query and a
sequence of inputs for keys and values. The sequence for the keys and values is
fixed for all queries.

Example
--------

    import torch

    from fast_transformers.recurrent.attention import \
        RecurrentCrossAttentionLayer, RecurrentCrossFullAttention

    att = RecurrentCrossAttentionLayer(RecurrentCrossFullAttention(), 16, 4)
    state = None
    x = torch.rand(8, 16)
    memory = torch.rand(8, 64, 16)
    for i in range(10):
        x, state = att(x, memory, memory, state=state)
"""

from .attention_layer import RecurrentCrossAttentionLayer
from .full_attention import RecurrentCrossFullAttention
from .linear_attention import RecurrentCrossLinearAttention
