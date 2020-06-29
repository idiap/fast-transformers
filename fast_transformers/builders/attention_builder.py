#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


class AttentionBuilder(object):
    """AttentionBuilder collects all the parameters for building attention
    implementations but does not contain the logic to build them.

    This class should be used as a mixin for other classes that imlement the
    building logic.
    """
    def __init__(self):
        # attention parameters
        self._attention_type = "full"
        self._softmax_temp = None
        self._linear_feature_map = None
        self._attention_dropout = 0.1
        self._clusters = 100
        self._iterations = 10
        self._bits = 32
        self._hash_bias = True
        self._topk = 32
        self._rounds = 4
        self._chunk_size = 32
        self._masked = False
        self._conditional_attention = False
        self._length_limit = 512

    @property
    def attention_type(self):
        """The attention implementation.

        One of {'full', 'linear', 'causal-linear', 'clustered',
                'improved-clustered', 'improved-causal', 'reformer',
                'exact-topk'}.
        """
        return self._attention_type

    @attention_type.setter
    def attention_type(self, val):
        attentions = ["full", "clustered", "improved-clustered",
                      "improved-causal", "linear", "causal-linear",
                      "reformer", "exact-topk"]
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

    @property
    def clusters(self):
        """Number of clusters for clustered attention variants."""
        return self._clusters

    @clusters.setter
    def clusters(self, val):
        self._clusters = val

    @property
    def bits(self):
        """Number of hash bits for clustered attention variants."""
        return self._bits

    @bits.setter
    def bits(self, val):
        self._bits = val

    @property
    def hash_bias(self):
        """If true, use eucliean distance hashing.
        If false, hashing is based on cosine distance"""
        return self._hash_bias

    @hash_bias.setter
    def hash_bias(self, val):
        self._hash_bias = bool(val)

    @property
    def iterations(self):
        """Number of Lloyd iterations for K-Means clustering step
        for the clustered attention variants"""
        return self._iterations

    @iterations.setter
    def iterations(self, val):
        self._iterations = val

    @property
    def topk(self):
        """Number of topk keys to be used for
        for the improved clustered attention"""
        return self._topk

    @topk.setter
    def topk(self, val):
        self._topk = val

    @property
    def rounds(self):
        """Number of hashing rounds to be used for for the reformer
        attention."""
        return self._rounds

    @rounds.setter
    def rounds(self, val):
        self._rounds = val

    @property
    def chunk_size(self):
        """Number of queries within each block of the reformer attention."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, val):
        self._chunk_size = val

    @property
    def masked(self):
        """If True the query doesn't attend to itself for reformer
        attention."""
        return self._masked

    @masked.setter
    def masked(self, val):
        self._masked = bool(val)

    @property
    def conditional_attention(self):
        """Use the 'attention_type' only for long sequences and use full
        otherwise."""
        return self._conditional_attention

    @conditional_attention.setter
    def conditional_attention(self, val):
        self._conditional_attention = bool(val)

    @property
    def length_limit(self):
        """Define the length limit that constitutes a long sequence to use with
        conditional_attention."""
        return self._length_limit

    @length_limit.setter
    def length_limit(self, val):
        self._length_limit = int(val)

