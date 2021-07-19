#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement transformer encoders and decoders that are going to be used with
different attention mechanisms.

In all cases the batch dimension is first and the sequence dimension is second.
"""

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from .events import EventDispatcher, IntermediateOutput
from .masking import FullMask, LengthMask


class TransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

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
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class TransformerEncoder(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoderLayer(Module):
    """The decoder layer from "Attention Is All You Need".

    Similar to the encoder layer, this layer implements the decoder that
    PyTorch implements but can be used with any attention implementation
    because it receives the attention layers as constructor arguments.

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
        super(TransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        """Apply the transformer decoder to the input x using the memory
        `memory`.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E should be the same as
               the d_model passed in the constructor.
            memory: The memory features of shape (N, L', E) where N is the
                    batch size, L' is the memory's sequence length (padded) and
                    E should be the same as the d_model.
            x_mask: An implementation of fast_transformers.masking.BaseMask
                    that encodes where each element of x can attend to in x.
                    Namely the self attention mask.
            x_length_mask: An implementation of a BaseMask that encodes how
                           many elements each sequence in the batch consists
                           of.
            memory_mask: An implementation of BaseMask that encodes where each
                         element of x can attend to in the memory. Namely the
                         cross attention mask.
            memory_length_mask: An implementation of a BaseMask that encodes how
                                many elements each memory sequence in the batch
                                consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
        x = self.norm1(x)

        # Secondly apply the cross attention and add it to the previous output
        x = x + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))

        # Finally run the fully connected part of the layer
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm3(x+y)


class TransformerDecoder(Module):
    """TransformerDecoder is little more than a sequence of transformer decoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ----------
        layers: list, TransformerDecoderLayer instances or instances that
                implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # Apply all the transformer decoders
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask,
                      memory_length_mask=memory_length_mask)
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x
