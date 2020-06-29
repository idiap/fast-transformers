#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement transformer encoders and decoders as RNNs that will be used with
different recurrent attention mechanisms.

In all cases there exists no sequence dimension and the shapes are batch x
heads x dims.

This module's interface is designed with the linear attention in mind. The
interface is subject to change given the implementation of other recurrent
attentions.
"""

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F


class RecurrentTransformerEncoderLayer(Module):
    """Attention to the previous inputs and feed forward with skip connections.

    This transformer encoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerEncoderLayer . The results should
    be identical given the same inputs and a lower triangular mask.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
    """
    def __init__(self, attention, d_model, n_heads, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(RecurrentTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, memory=None):
        """Apply the transformer encoder to the input x using the provided
        memory.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            memory: The memory can vary depending on the attention implementation
        """
        # Run the self attention and add it to the input
        x2, memory = self.attention(x, x, x, memory)
        x = x + self.dropout(x2)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y), memory


class RecurrentTransformerEncoder(Module):
    """RecurrentTransformerEncoder is a sequence of
    RecurrentTransformerEncoderLayer instances.

    RecurrentTransformerEncoder keeps a separate memory per
    RecurrentTransformerEncoderLayer.

    Arguments
    ---------
        layers: list, RecurrentTransformerEncoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
    """
    def __init__(self, layers, norm_layer=None):
        super(RecurrentTransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, memory=None):
        """Apply all recurrent transformer layers to the input x using the
        provided memory.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor of each recurrent
               transformer encoder layer
            memory: A list of objects to be passed to each recurrent
                    transformer encoder layer
        """
        # Initialize the memory to None if not given
        if memory is None:
            memory = [None]*len(self.layers)

        # Apply all the transformers
        for i, layer in enumerate(self.layers):
            x, m = layer(x, memory[i])
            memory[i] = m

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, memory
