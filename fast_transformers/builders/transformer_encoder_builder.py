#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the TransformerEncoderBuilder that constructs transformer encoders
with various attention mechanisms.

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

from functools import partial

from torch.nn import LayerNorm

from .base import BaseTransformerBuilder
from .common_encoder_builder import CommonEncoderBuilder
from .attention_builder import AttentionBuilder
from ..attention import AttentionLayer, FullAttention, \
    LinearAttention, CausalLinearAttention, \
    ClusteredAttention, ImprovedClusteredAttention, \
    ImprovedClusteredCausalAttention, \
    ReformerAttention, ConditionalFullAttention, \
    ExactTopKAttention
from ..transformers import TransformerEncoder, \
    TransformerEncoderLayer


class TransformerEncoderBuilder(BaseTransformerBuilder, CommonEncoderBuilder,
                                AttentionBuilder):
    """TransformerEncoderBuilder builds transformer encoders (duh).

    This means that the module returned is going to be an instance of
    fast_transformer.transformers.TransformerEncoder.
    """
    def __init__(self):
        CommonEncoderBuilder.__init__(self)
        AttentionBuilder.__init__(self)

    def __repr__(self):
        return (
            "TransformerEncoderBuilder.from_kwargs(\n"
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
            "    attention_dropout={!r},\n"
            "    clusters={!r},\n"
            "    bits={!r},\n"
            "    hash_bias={!r},\n"
            "    iterations={!r},\n"
            "    topk={!r},\n"
            "    chunk_size={!r},\n"
            "    rounds={!r},\n"
            "    masked={!r},\n"
            "    conditional_attention={!r},\n"
            "    length_limit={!r}\n"
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
            self.attention_dropout,
            self.clusters,
            self.bits,
            self.hash_bias,
            self.iterations,
            self.topk,
            self.chunk_size,
            self.rounds,
            self.masked,
            self.conditional_attention,
            self.length_limit
        )

    def _get_attention(self):
        attentions = {
            "full": partial(
                FullAttention,
                softmax_temp=self.softmax_temp,
                dropout_rate=self.attention_dropout
            ),
            "clustered": partial(
                ClusteredAttention,
                self.clusters,
                self.iterations,
                self.bits,
                self.hash_bias,
                self.softmax_temp,
                self.attention_dropout
            ),
            "improved-clustered": partial(
                ImprovedClusteredAttention,
                self.clusters,
                self.iterations,
                self.bits,
                self.hash_bias,
                self.topk,
                self.softmax_temp,
                self.attention_dropout
            ),
            "improved-causal": partial(
                ImprovedClusteredCausalAttention,
                self.clusters,
                self.iterations,
                self.bits,
                self.hash_bias,
                self.topk,
                self.softmax_temp,
                self.attention_dropout
            ),
            "reformer": partial(
                ReformerAttention,
                self.chunk_size,
                self.bits,
                self.rounds,
                self.masked,
                self.softmax_temp,
                self.attention_dropout
            ),
            "exact-topk": partial(
                ExactTopKAttention,
                self.topk,
                self.softmax_temp,
                self.attention_dropout
            ),
            "linear": partial(LinearAttention, self.linear_feature_map),
            "causal-linear": partial(
                CausalLinearAttention,
                self.linear_feature_map
            )
        }
        attention = attentions[self.attention_type]()

        if self.conditional_attention:
            attention = ConditionalFullAttention(
                attention,
                self.length_limit,
                self.softmax_temp,
                self.attention_dropout
            )

        return attention

    def get(self):
        model_dimensions = self.value_dimensions*self.n_heads
        return TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
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
