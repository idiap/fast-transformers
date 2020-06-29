#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implementations of different types of autoregressive attention
mechanisms."""

from .attention_layer import RecurrentAttentionLayer
from .full_attention import RecurrentFullAttention
from .linear_attention import RecurrentLinearAttention
