#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""This module implements builders that simplify building complex transformer
architectures with different attention mechanisms.

The main idea is to facilitate the construction of various attention layers and
transformer encoder layers and simplify their assembly into one transformer
module. It also allows for flexibility in the scripts as many builder
parameters can correspond 1-1 with command line arguments.

Example usage:

    builder = TransformerEncoderBuilder()
    builder.n_layers = 12
    builder.n_heads = 8
    builder.feed_forward_dimensions = 1024
    builder.query_dimensions = 64
    builder.value_dimensions = 64
    builder.dropout = 0.1
    builder.attention_dropout = 0.1
    builder.attention_type = "linear"
    transformer = builder.get()
"""

__all__ = [
    "AttentionBuilder",
    "RecurrentAttentionBuilder",
    "RecurrentCrossAttentionBuilder"
]

# Import the attention implementations so that they register themselves with
# the builder. Attention implementations external to the library should be
# imported before using the builders.
#
# TODO: Should this behaviour change? Namely, should all attention
#       implementations be imported in order to be useable? This also allows
#       using the library even partially built, for instance.
from ..attention import \
    FullAttention, \
    LinearAttention, CausalLinearAttention, \
    ClusteredAttention, ImprovedClusteredAttention, \
    ReformerAttention, \
    ExactTopKAttention, ImprovedClusteredCausalAttention, \
    ConditionalFullAttention
del FullAttention, \
    LinearAttention, CausalLinearAttention, \
    ClusteredAttention, ImprovedClusteredAttention, \
    ReformerAttention, \
    ExactTopKAttention, ImprovedClusteredCausalAttention, \
    ConditionalFullAttention


from .attention_builders import \
    AttentionBuilder, \
    RecurrentAttentionBuilder, \
    RecurrentCrossAttentionBuilder

from .transformer_builders import \
    TransformerEncoderBuilder, \
    RecurrentEncoderBuilder, \
    TransformerDecoderBuilder, \
    RecurrentDecoderBuilder
