#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


class CommonEncoderBuilder(object):
    """CommonEncoderBuilder collects all the parameters for building a
    transformer encoder but does not contain the logic to build one or the
    logic to collect the attention parameters.

    This class should be used as a mixin for other classes that implement the
    building logic.
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

