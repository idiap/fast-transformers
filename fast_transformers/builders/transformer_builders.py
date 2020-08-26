#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Build complex transformer architectures for inference or training easily."""

from torch.nn import LayerNorm

from ..attention import AttentionLayer
from ..transformers import TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer
from ..recurrent.attention import \
    RecurrentAttentionLayer, \
    RecurrentCrossAttentionLayer
from ..recurrent.transformers import \
    RecurrentTransformerEncoder, RecurrentTransformerEncoderLayer, \
    RecurrentTransformerDecoder, RecurrentTransformerDecoderLayer
from .base import BaseBuilder
from .attention_builders import AttentionBuilder, RecurrentAttentionBuilder, \
    RecurrentCrossAttentionBuilder


class BaseTransformerBuilder(BaseBuilder):
    """Contains all the parameters for building a transformer other than the
    attention part.

    Classes extending the BaseTransformerBuilder should implement the `get()`
    method that actually builds the transformer.
    """
    def __init__(self):
        # transformer parameters
        self._n_layers = 4
        self._n_heads = 4
        self._d_query = 64
        self._d_value = 64
        self._d_ff = 1024
        self._dropout = 0.1
        self._activation = "relu"
        self._final_norm = True

    @property
    def n_layers(self):
        """The number of transformer layers."""
        return self._n_layers

    @n_layers.setter
    def n_layers(self, val):
        self._n_layers = val

    @property
    def n_heads(self):
        """The number of heads in each transformer layer."""
        return self._n_heads

    @n_heads.setter
    def n_heads(self, val):
        self._n_heads = val

    @property
    def feed_forward_dimensions(self):
        """The dimensions of the fully connected layer in the transformer
        layers."""
        return self._d_ff

    @feed_forward_dimensions.setter
    def feed_forward_dimensions(self, val):
        self._d_ff = val

    @property
    def query_dimensions(self):
        """The dimensions of the queries and keys in each attention layer."""
        return self._d_query

    @query_dimensions.setter
    def query_dimensions(self, val):
        self._d_query = val

    @property
    def value_dimensions(self):
        """The dimensions of the values in each attention layer."""
        return self._d_value

    @value_dimensions.setter
    def value_dimensions(self, val):
        self._d_value = val

    @property
    def dropout(self):
        """The dropout rate to be applied in the transformer encoder layer."""
        return self._dropout

    @dropout.setter
    def dropout(self, val):
        self._dropout = val

    @property
    def activation(self):
        """The activation function for the transformer layer.

        One of {'relu', 'gelu'}.
        """
        return self._activation

    @activation.setter
    def activation(self, val):
        activations = ["relu", "gelu"]
        if val not in activations:
            raise ValueError(("{!r} is not one of the availabel activation "
                              "types {!r}").format(val, activations))
        self._activation = val

    @property
    def final_normalization(self):
        """Whether to add LayerNorm as the final layer of the
        TransformerEncoder."""
        return self._final_norm

    @final_normalization.setter
    def final_normalization(self, val):
        self._final_norm = bool(val)

    def get(self):
        """Build the transformer and return it."""
        raise NotImplementedError()


class BaseTransformerEncoderBuilder(BaseTransformerBuilder):
    """Implement the logic of building a transformer encoder but leave the
    specific layers open for changing by the inheriting classes. This allows us
    to reuse the logic for creating both the TransformerEncoder and the
    RecurrentTransformerEncoder.

    Inheriting classes should implement the following:

    - _get_attention_builder()
    - _get_attention_layer_class()
    - _get_encoder_class()
    - _get_encoder_layer_class()
    """
    def __init__(self):
        super(BaseTransformerEncoderBuilder, self).__init__()
        self._attention_builder = self._get_attention_builder()
        self._attention_type = "full"

    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        raise NotImplementedError()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        raise NotImplementedError()

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        raise NotImplementedError()

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        raise NotImplementedError()

    @property
    def attention(self):
        """The attention builder instance."""
        return self._attention_builder

    @property
    def attention_type(self):
        """The attention implementation chosen."""
        return self._attention_type

    @attention_type.setter
    def attention_type(self, val):
        if not self._attention_builder.validate_attention_type(val):
            raise ValueError(("{!r} is not an available attention "
                              "type").format(val))
        self._attention_type = val

    def __setattr__(self, key, val):
        # "protected" attributes are settable (probably from withing the class)
        if key[0] == "_":
            return super().__setattr__(key, val)

        # Existing attributes are settable
        if hasattr(self, key):
            return super().__setattr__(key, val)

        # Non-existing "public" attributes may be attention parameters
        setattr(self._attention_builder, key, val)

    def get(self):
        """Build the transformer and return it."""
        # Extract into local variables the classes to be used
        Encoder = self._get_encoder_class()
        EncoderLayer = self._get_encoder_layer_class()
        Attention = self._get_attention_layer_class()

        model_dimensions = self.value_dimensions*self.n_heads
        return Encoder(
            [
                EncoderLayer(
                    Attention(
                        self.attention.get(self.attention_type),
                        model_dimensions,
                        self.n_heads,
                        d_keys=self.query_dimensions,
                        d_values=self.value_dimensions
                    ),
                    model_dimensions,
                    self.n_heads,  # Should be removed (see #7)
                    self.feed_forward_dimensions,
                    self.dropout,
                    self.activation
                )
                for _ in range(self.n_layers)
            ],
            (LayerNorm(model_dimensions) if self.final_normalization else None)
        )


class TransformerEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder for training or processing of
    sequences all elements at a time.

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
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return AttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer


class RecurrentEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a transformer encoder for autoregressive processing of sequences.

    Example usage:

        builder = RecurrentEncoderBuilder()
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
    def _get_attention_builder(self):
        """Return an attention builder for recurrent attention."""
        return RecurrentAttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the recurrent layer that projects queries keys
        and values."""
        return RecurrentAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the recurrent transformer encoder."""
        return RecurrentTransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the recurrent transformer encoder layer."""
        return RecurrentTransformerEncoderLayer


class BaseTransformerDecoderBuilder(BaseTransformerBuilder):
    """Similar to BaseTransformerEncoderBuilder implement the logic of
    building the transformer decoder without defining concrete layers.

    Inheriting classes should implement the following:

    - _get_self_attention_builder() and _get_cross_attention_builder()
    - _get_self_attention_layer_class() and _get_cross_attention_layer_class()
    - _get_decoder_class()
    - _get_decoder_layer_class()
    """
    def __init__(self):
        super(BaseTransformerDecoderBuilder, self).__init__()
        self._self_attention_builder = self._get_self_attention_builder()
        self._cross_attention_builder = self._get_cross_attention_builder()
        self._self_attention_type = "full"
        self._cross_attention_type = "full"

    def _get_self_attention_builder(self):
        """Return an instance of attention builder."""
        raise NotImplementedError()

    def _get_cross_attention_builder(self):
        """Return an instance of attention builder."""
        raise NotImplementedError()

    def _get_self_attention_layer_class(self):
        """Return a class to project the queries, keys and values to
        multi-head versions."""
        raise NotImplementedError()

    def _get_cross_attention_layer_class(self):
        """Return a class to project the queries, keys and values to
        multi-head versions."""
        raise NotImplementedError()

    def _get_decoder_class(self):
        """Return the class for the transformer decoder."""
        raise NotImplementedError()

    def _get_decoder_layer_class(self):
        """Return the class for the transformer decoder layer."""
        raise NotImplementedError()

    @property
    def self_attention(self):
        """The attention builder instance that will be used for the self
        attention modules."""
        return self._self_attention_builder

    @property
    def self_attention_type(self):
        """The attention implementation used for self attention."""
        return self._self_attention_type

    @self_attention_type.setter
    def self_attention_type(self, val):
        if not self._self_attention_builder.validate_attention_type(val):
            raise ValueError(("{!r} is not an available self attention "
                              "type").format(val))
        self._self_attention_type = val

    @property
    def cross_attention(self):
        """The attention builder instance that will be used for the cross
        attention modules."""
        return self._cross_attention_builder

    @property
    def cross_attention_type(self):
        """The attention implementation used for cross attention."""
        return self._cross_attention_type

    @cross_attention_type.setter
    def cross_attention_type(self, val):
        if not self._cross_attention_builder.validate_attention_type(val):
            raise ValueError(("{!r} is not an available cross attention "
                              "type").format(val))
        self._cross_attention_type = val

    def __setattr__(self, key, val):
        # "protected" attributes are settable (probably from withing the class)
        if key[0] == "_":
            return super().__setattr__(key, val)

        # Existing attributes are settable
        if hasattr(self, key):
            return super().__setattr__(key, val)

        # Non-existing "public" attributes may be attention parameters
        setattr(self._self_attention_builder, key, val)
        setattr(self._cross_attention_builder, key, val)

    def get(self):
        """Build the transformer and return it."""
        # Extract into local variables the classes to be used
        Decoder = self._get_decoder_class()
        DecoderLayer = self._get_decoder_layer_class()
        SelfAttention = self._get_self_attention_layer_class()
        CrossAttention = self._get_cross_attention_layer_class()

        model_dimensions = self.value_dimensions*self.n_heads
        return Decoder(
            [
                DecoderLayer(
                    SelfAttention(
                        self.self_attention.get(self.self_attention_type),
                        model_dimensions,
                        self.n_heads,
                        d_keys=self.query_dimensions,
                        d_values=self.value_dimensions
                    ),
                    CrossAttention(
                        self.cross_attention.get(self.cross_attention_type),
                        model_dimensions,
                        self.n_heads,
                        d_keys=self.query_dimensions,
                        d_values=self.value_dimensions
                    ),
                    model_dimensions,
                    self.feed_forward_dimensions,
                    self.dropout,
                    self.activation
                )
                for _ in range(self.n_layers)
            ],
            (LayerNorm(model_dimensions) if self.final_normalization else None)
        )


class TransformerDecoderBuilder(BaseTransformerDecoderBuilder):
    """Build a transformer decoder for training or processing of sequences all
    elements at a time.

    Example usage:

        builder = TransformerDecoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.self_attention_type = "full"
        builder.cross_attention_type = "full"
        transformer = builder.get()
    """
    def _get_self_attention_builder(self):
        """Return an attention builder for creating non-recurrent attention
        variants."""
        return AttentionBuilder()

    def _get_cross_attention_builder(self):
        """Return an attention builder for creating non-recurrent attention
        variants."""
        return AttentionBuilder()

    def _get_self_attention_layer_class(self):
        """Return the non-recurrent attention layer to project queries, keys
        and values."""
        return AttentionLayer

    def _get_cross_attention_layer_class(self):
        """Return the non-recurrent attention layer to project queries, keys
        and values."""
        return AttentionLayer

    def _get_decoder_class(self):
        """Return the transformer decoder class."""
        return TransformerDecoder

    def _get_decoder_layer_class(self):
        """Return the transformer decoder layer class."""
        return TransformerDecoderLayer


class RecurrentDecoderBuilder(BaseTransformerDecoderBuilder):
    """Build a transformer decoder for processing of sequences in
    autoregressive fashion.

    Example usage:

        builder = RecurrentDecoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.self_attention_type = "full"
        builder.cross_attention_type = "full"
        transformer = builder.get()
    """
    def _get_self_attention_builder(self):
        """Return an attention builder for creating non-recurrent attention
        variants."""
        return RecurrentAttentionBuilder()

    def _get_cross_attention_builder(self):
        """Return an attention builder for creating non-recurrent attention
        variants."""
        return RecurrentCrossAttentionBuilder()

    def _get_self_attention_layer_class(self):
        """Return the non-recurrent attention layer to project queries, keys
        and values."""
        return RecurrentAttentionLayer

    def _get_cross_attention_layer_class(self):
        """Return the non-recurrent attention layer to project queries, keys
        and values."""
        return RecurrentCrossAttentionLayer

    def _get_decoder_class(self):
        """Return the transformer decoder class."""
        return RecurrentTransformerDecoder

    def _get_decoder_layer_class(self):
        """Return the transformer decoder layer class."""
        return RecurrentTransformerDecoderLayer
