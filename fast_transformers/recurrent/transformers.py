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

import warnings

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from ..events import EventDispatcher, IntermediateOutput
from ..masking import LengthMask
from ._utils import check_state


class RecurrentTransformerEncoderLayer(Module):
    """Attention to the previous inputs and feed forward with skip connections.

    This transformer encoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerEncoderLayer . The results should
    be identical given the same inputs and a lower triangular mask.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(RecurrentTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None, memory=None):
        """Apply the transformer encoder to the input x using the provided
        memory.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            state: The state can vary depending on the attention implementation
            memory: **Deprecated** name for the state argument
        """
        # Normalize the state name
        state = check_state(state, memory)

        # Run the self attention and add it to the input
        x2, state = self.attention(x, x, x, state)
        x = x + self.dropout(x2)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y), state


class RecurrentTransformerEncoder(Module):
    """RecurrentTransformerEncoder is a sequence of
    RecurrentTransformerEncoderLayer instances.

    RecurrentTransformerEncoder keeps a separate state per
    RecurrentTransformerEncoderLayer.

    Arguments
    ---------
        layers: list, RecurrentTransformerEncoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(RecurrentTransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None, memory=None):
        """Apply all recurrent transformer layers to the input x using the
        provided state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor of each recurrent
               transformer encoder layer
            state: A list of objects to be passed to each recurrent
                   transformer encoder layer
            memory: **Deprecated** name for the state argument
        """
        # Initialize the memory to None if not given
        state = check_state(state, memory)
        if state is None:
            state = [None]*len(self.layers)

        # Apply all the transformers
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state[i])
            state[i] = s
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, state


class RecurrentTransformerDecoderLayer(Module):
    """Attention to the previous inputs and a preprocessed memory.

    This transformer decoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerDecoderLayer . The results should
    be identical given the same inputs and a lower triangular mask for x_mask.

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
                        given as a nn.Module
        cross_attention: The attention implementation to use for cross
                         attention given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", event_dispatcher=""):
        super(RecurrentTransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, memory_length_mask=None, state=None):
        """Apply the transformer decoder to the input x and also attend to
        memory.

        Note the memory mask is assumed to be a full mask.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            memory: A sequence of features (N, S, E) that the input will attend
                    to. S is the sequence length and E is the same as for x.
            memory_length_mask: An implementation of a BaseMask that encodes
                                how many elements each memory sequence in the
                                batch consists of.
            state: The state varies depending on the attention implementations
                   but it allows for recurrent implementation.
        """
        # Normalize the mask
        N = x.shape[0]
        L = memory.shape[1]
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Extract the individual states for the self attention and the cross
        # attention
        self_state, cross_state = state or [None, None]

        # First apply the self attention and add it to the input
        x2, self_state = self.self_attention(x, x, x, state=self_state)
        x = self.norm1(x + self.dropout(x2))

        # Secondly apply the cross attention and add it to the previous output
        x2, cross_state = self.cross_attention(
            x, memory, memory, memory_length_mask, state=cross_state
        )
        x = self.norm2(x + self.dropout(x2))

        # Finally run the fully connected part of the layer
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm3(x+y), [self_state, cross_state]


class RecurrentTransformerDecoder(Module):
    """RecurrentTransformerDecoder is little more than a sequence of
    RecurrentTransformerDecoderLayer instances.

    Simlar to the recurrent encoder a separate state is kept per decoder layer.

    Arguments
    ---------
        layers: list, RecurrentTransformerDecoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(RecurrentTransformerDecoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, memory_length_mask=None, state=None):
        """Apply all recurrent transformer layers to the input x using the
        provided state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            memory: A sequence of features (N, S, E) that the input will attend
                    to. S is the sequence length and E is the same as for x.
            memory_length_mask: An implementation of a BaseMask that encodes
                                how many elements each memory sequence in the
                                batch consists of
            state: A list of objects to be passed to each recurrent
                   transformer decoder layer
        """
        # Initialize the state to None if not given
        if state is None:
            state = [None]*len(self.layers)

        # Apply all the transformers
        for i, layer in enumerate(self.layers):
            x, s = layer(x, memory, memory_length_mask=memory_length_mask,
                         state=state[i])
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))
            state[i] = s

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, state
