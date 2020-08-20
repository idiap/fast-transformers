#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""The parameters objects hold the arguments necessary for constructing
transformer and attention modules but not the logic. This way a builder can
utilize multiple instances of parameter objects as needed for constructing
different parts of a transformer."""


class BaseParameters(object):
    """A utility base class for parameter objects."""
    def __repr__(self):
        props = [
            k for k in dir(self)
            if not k.startswith("_") and not k.startswith("from_")
        ]
        args = "\n".join([
            "    {}={!r},".format(p, getattr(self, p))
            for p in props
        ])
        return ("{}.from_kwargs(\n" + args[:-1] + "\n)").format(
            self.__class__.__name__
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Construct a Parameters object and set all the keyword arguments as
        parameters.

        The keyword argument strict is passed to
        BaseParameters.from_dictionary separately.

        See BaseParameters.from_dictionary().
        """
        strict = kwargs.pop("strict", True)
        return cls.from_dictionary(kwargs, strict=strict)

    @classmethod
    def from_namespace(cls, args, strict=False):
        """Construct a Parameters object from an argparse Namespace.

        To be used for building transformers from command line arguments.

        See BaseParameters.from_dictionary().
        """
        return cls.from_dictionary(vars(args), strict=strict)

    @classmethod
    def from_dictionary(cls, dictionary, strict=True):
        """Construct a Parameters objects and set all the parameters in the
        dictionary.

        Given a dictionary

            d = {"foo": "bar"}

        then

            params = TransformerParameters.from_dictionary(d)

        is equivalent to

            params = TransformerParameters()
            params.foo = "bar"

        Arguments
        ---------
            dictionary: A dictionary of parameters to set to the builder.
            strict: bool, If a key is not a parameter and strict is set to True
                    then a ValueError is raised, otherwise that dictionary key
                    is ignored (default: True)
        """
        params = cls()
        for k, v in dictionary.items():
            if not hasattr(params, k):
                if strict:
                    raise ValueError(("The params object has no "
                                      "parameter {!r}").format(k))
                else:
                    continue
            setattr(params, k, v)
        return params


class TransformerParameters(BaseParameters):
    """"""
    def __init__(self):
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


class AttentionParameters(BaseParameters):
    """"""
    def __init__(self, acceptable_types=None):
        self._acceptable_types = acceptable_types or {
            "full", "linear", "causal-linear", "clustered",
            "improved-clustered", "improved-causal", "reformer", "exact-topk"
        }
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

        Must be a string that is defined in the acceptable types passed in the
        constructor.
        """
        return self._attention_type

    @attention_type.setter
    def attention_type(self, val):
        attentions = ["full", "clustered", "improved-clustered",
                      "improved-causal", "linear", "causal-linear",
                      "reformer", "exact-topk"]
        if val not in self._acceptable_types:
            raise ValueError(("{!r} is not one of the available attention "
                              "types {!r}").format(val, self._acceptable_types))
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
