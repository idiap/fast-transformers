#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Build complex transformer architectures for inference or training easily."""

from torch.nn import LayerNorm

from ..attention import AttentionLayer
from ..transformers import TransformerEncoder, TransformerEncoderLayer
from ..recurrent.attention import RecurrentAttentionLayer
from ..recurrent.transformers import \
    RecurrentTransformerEncoder, RecurrentTransformerEncoderLayer
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


class _BaseTransformerEncoderBuilder(BaseTransformerBuilder):
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
        super(_BaseTransformerEncoderBuilder, self).__init__()
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
        # Extract the classes to be used in local variables
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


class TransformerEncoderBuilder(_BaseTransformerEncoderBuilder):
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


class RecurrentEncoderBuilder(_BaseTransformerEncoderBuilder):
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
        """Return an instance of the appropriate attention builder."""
        return RecurrentAttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RecurrentAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return RecurrentTransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return RecurrentTransformerEncoderLayer
