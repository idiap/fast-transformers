#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implementations of different types of autoregressive attention
mechanisms for self attention and cross attention."""

from .self_attention.attention_layer import RecurrentAttentionLayer
from .self_attention.full_attention import RecurrentFullAttention
from .self_attention.linear_attention import RecurrentLinearAttention

from .cross_attention.attention_layer import RecurrentCrossAttentionLayer
from .cross_attention.full_attention import RecurrentCrossFullAttention
from .cross_attention.linear_attention import RecurrentCrossLinearAttention
