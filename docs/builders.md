Builders
========

The builders module takes care of simplifying the construction of transformer
networks. The following example showcases how simple it is to create a
transformer encoder using the [TransformerEncoderBuilder][1].

```python
import torch

# Building without a builder
from fast_transformers.transformers import TransformerEncoder, \
    TransformerEncoderLayer
from fast_transformers.attention import AttentionLayer, FullAttention

bert = TransformerEncoder(
    [
        TransformerEncoderLayer(
            AttentionLayer(FullAttention(), 768, 12),
            768,
            12,
            activation="gelu"
        ) for l in range(12)
    ],
    norm_layer=torch.nn.LayerNorm(768)
)

# Building with a builder
from fast_transformers.builders import TransformerEncoderBuilder
bert = TransformerEncoderBuilder.from_kwargs(
    attention_type="full",
    n_layers=12,
    n_heads=12,
    feed_forward_dimensions=768*4,
    query_dimensions=768,
    activation="gelu"
)
```

Although it seems that the creation of a transformer is as simple with and
without the builder, it becomes apparent that changing the creation logic with
the builder is significantly easier. For instance, the `attention_type` can be
read from a configuration file or from command line arguments.
The rest of this page describes the API of the builders.

Builder API
------------

The interface for all the builders is a simple method `get()` without any
arguments that returns a PyTorch module that implements a transformer.

All the parameters of the builders are simple python properties that can be set
after the creation of the builder object.

```python
builder = ...                          # create a builder

builder.parameter = value              # set a parameter
builder.other_parameter = other_value  # and another parameter
transformer = builder.get()            # construct the transformer

builder.parameter = changed_value      # change a parameter
other_transformer = builder.get()      # construct another transformer
```

The [BaseBuilder][2] provides helper static methods that make it simpler to set
multiple builder arguments at once from configuration files or command line
arguments.

```python
from_dictionary(dictionary, strict=True)
```

Construct a builder and set all the parameters in the dictionary. If `strict`
is set to True then throw a ValueError in case a dictionary key does not
correspond to a builder parameter.

```python
from_kwargs(**kwargs)
```

Construct a builder and set all the keyword arguments as builder parameters.

```python
from_namespace(args, strict=False)
```

Construct a builder from an argument list returned by the python argparse
module. If `strict` is set to True then throw a ValueError in case an argument
does not correspond to a builder parameter.

Transformer Builders
--------------------

There exist the following transformer builders for creating encoder and decoder
architectures for inference and training:

* [**TransformerEncoderBuilder**][1] builds instances of [TransformerEncoder][3]
* [**TransformerDecoderBuilder**][1] builds instances of [TransformerDecoder][8]
* [**RecurrentEncoderBuilder**][1] builds instances of [RecurrentTransformerEncoder][4]
* [**RecurrentDecoderBuilder**][1] builds instances of [RecurrentTransformerDecoder][9]

Attention Builders
------------------

[Attention builders][5] simplify the construction of the various attention
modules and allow for plugin-like extension mechanisms when creating new
attention implementations.

Their API is the same as the transformer builders, namely they accept
attributes as parameters and then calling `get(attention_type: str)` constructs
an `nn.Module` that implements an attention layer.

```python
from fast_transformers.builders import AttentionBuilder

builder = AttentionBuilder.from_kwargs(
    attention_dropout=0.1,                   # used by softmax attention
    softmax_temp=1.,                         # used by softmax attention
    feature_map=lambda x: (x>0).float() * x  # used by linear
)
softmax = builder.get("full")
linear = builder.get("linear")
```

The library provides the following attention builders that create the
correspondingly named attention modules.

* AttentionBuilder
* RecurrentAttentionBuilder
* RecurrentCrossAttentionBuilder

### Attention composition

The attention builders allow for *attention composition* through a simple
convention of the `attention_type` parameter. Attention composition allows the
creation of an attention layer that accepts one or more attention layers as a
parameters. An example of this pattern is the [ConditionalFullAttention][6] that
performs full softmax attention when the sequence length is small and delegates
to another attention type when the sequence length becomes large.

The following example code creates an attention layer that uses [improved
clustered attention][7] for sequences larger than 512 elements and full softmax
attention otherwise.

```python
builder = AttentionBuilder.from_kwargs(
    attention_dropout=0.1,  # used by all
    softmax_temp=0.125,
    topk=32,                # used by improved clustered
    clusters=256,
    bits=32,
    length_limit=512        # used by conditional attention
)
attention = builder.get("conditional-full:improved-clustered")
```

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>Attention layers that are designed for composition cannot be used
    standalone. For instance <code>conditional-full</code> is not a valid
    attention type by itsself.</p>
</div>

### Attention Registry

The attention builders allow the dynamic registering of attention
implementations through an [attention registry][10]. There are three
registries, one for each available builder. You can find plenty of usage
examples in the provided attention implementations (e.g. [FullAttention][11]).

This should only concern developers of new attention implementations and a
simple example can be found in the [custom attention
layer](custom_attention_layer.md) section of the docs.

[1]: /api_docs/fast_transformers/builders/transformer_builders.html
[2]: /api_docs/fast_transformers/builders/base.html
[3]: /api_docs/fast_transformers/transformers.html#fast_transformers.transformers.TransformerEncoder
[4]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoder
[5]: /api_docs/fast_transformers/builders/attention_builders.html
[6]: /api_docs/fast_transformers/attention/conditional_full_attention.html
[7]: /api_docs/fast_transformers/attention/improved_clustered_attention.html
[8]: /api_docs/fast_transformers/transformers.html#fast_transformers.transformers.TransformerDecoder
[9]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerDecoder
[10]: /api_docs/fast_transformers/attention_registry/index.html
[11]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/full_attention.py
