#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement a builder for transformers implemented as recurrent networks
mostly for autoregressive inference.

Example usage:

    builder = RecurrentEncoderBuilder()
    builder.n_layers = 12
    builder.n_heads = 8
    builder.feed_forward_dimensions = 1024
    builder.query_dimensions = 64
    builder.value_dimensions = 64
    builder.attention_type = "linear"
    transformer = builder.get()
"""

from functools import partial

from torch.nn import LayerNorm

from .base import BaseTransformerBuilder
from .common_encoder_builder import CommonEncoderBuilder
from .recurrent_attention_builder import RecurrentAttentionBuilder
from ..recurrent.attention import RecurrentAttentionLayer, \
    RecurrentFullAttention, RecurrentLinearAttention
from ..recurrent.transformers import RecurrentTransformerEncoder, \
    RecurrentTransformerEncoderLayer


class RecurrentEncoderBuilder(BaseTransformerBuilder, CommonEncoderBuilder,
                              RecurrentAttentionBuilder):
    """Build recurrent transformers for autoregressive inference.

    This means that the module returned is going to be an instance of
    fast_transformer.recurrent.transformers.RecurrentTransformerEncoder .
    """
    def __init__(self):
        CommonEncoderBuilder.__init__(self)
        RecurrentAttentionBuilder.__init__(self)

    def __repr__(self):
        return (
            "RecurrentEncoderBuilder.from_kwargs(\n"
            "    n_layers={!r},\n"
            "    n_heads={!r},\n"
            "    feed_forward_dimensions={!r},\n"
            "    query_dimensions={!r},\n"
            "    value_dimensions={!r},\n"
            "    dropout={!r},\n"
            "    activation={!r},\n"
            "    final_normalization={!r},\n"
            "    attention_type={!r},\n"
            "    softmax_temp={!r},\n"
            "    linear_feature_map={!r},\n"
            "    attention_dropout={!r}\n"
            ")"
        ).format(
            self.n_layers,
            self.n_heads,
            self.feed_forward_dimensions,
            self.query_dimensions,
            self.value_dimensions,
            self.dropout,
            self.activation,
            self.final_normalization,
            self.attention_type,
            self.softmax_temp,
            self.linear_feature_map,
            self.attention_dropout
        )

    def _get_attention(self):
        full = partial(
            RecurrentFullAttention,
            softmax_temp=self.softmax_temp,
            dropout_rate=self.attention_dropout
        )
        linear = partial(RecurrentLinearAttention, self._linear_feature_map)

        attentions = {
            "full": full,
            "linear": linear,
            "causal-linear": linear
        }
        
        return attentions[self.attention_type]()

    def get(self):
        model_dimensions = self.value_dimensions*self.n_heads
        return RecurrentTransformerEncoder(
            [
                RecurrentTransformerEncoderLayer(
                    RecurrentAttentionLayer(
                        self._get_attention(),
                        model_dimensions,
                        self.n_heads,
                        d_keys=self.query_dimensions,
                        d_values=self.value_dimensions
                    ),
                    model_dimensions,
                    self.n_heads,
                    self.feed_forward_dimensions,
                    self.dropout,
                    self.activation
                )
                for _ in range(self.n_layers)
            ],
            (LayerNorm(model_dimensions) if self._final_norm else None)
        )
