#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


class RecurrentAttentionBuilder(object):
    """RecurrentAttentionBuilder collects all the parameters for building
    recurrent attention implementations but does not contain the logic to build
    them.

    This class should be used as a mixin for other classes that imlement the
    building logic.
    """
    def __init__(self):
        # attention parameters
        self._attention_type = "full"
        self._softmax_temp = None
        self._linear_feature_map = None
        self._attention_dropout = 0.1

    @property
    def attention_type(self):
        """The attention implementation.

        One of {'full', 'linear', 'causal-linear'}.
        """
        return self._attention_type

    @attention_type.setter
    def attention_type(self, val):
        attentions = ["full", "linear", "causal-linear"]
        if val not in attentions:
            raise ValueError(("{!r} is not one of the available attention "
                              "types {!r}").format(val, attentions))
        self._attention_type = val

    @property
    def softmax_temp(self):
        """The temperature for the softmax in the attention.

        If it is set to None then 1/sqrt(builder.query_dimension) will be used.
        """
        return self._softmax_temp

    @softmax_temp.setter
    def softmax_temp(self, val):
        self._softmax_temp = val

    @property
    def linear_feature_map(self):
        """The feature map to use with linear attention.

        It should be a callable or None in which case elu(x)+1 will be used.
        """
        return self._linear_feature_map

    @linear_feature_map.setter
    def linear_feature_map(self, val):
        if val is not None and not callable(val):
            raise ValueError(("The linear_feature_map should be a callable or "
                              "None"))
        self._linear_feature_map = val

    @property
    def attention_dropout(self):
        """The dropout rate for the attention matrix."""
        return self._attention_dropout

    @attention_dropout.setter
    def attention_dropout(self, val):
        self._attention_dropout = val
