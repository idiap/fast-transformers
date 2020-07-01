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

Transformer Builder API
-----------------------

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

The [BaseTransformerBuilder][2] provides helper static methods that make it
simpler to set multiple builder arguments at once from configuration files or
command line arguments.

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

Available Builders
------------------

* **[TransformerEncoderBuilder][1]** constructs instances of [TransformerEncoder][3]
* **[RecurrentEncoderBuilder][5]** constructs instances of [RecurrentTransformerEncoder][4]


[1]: /api_docs/fast_transformers/builders/transformer_encoder_builder.html
[2]: /api_docs/fast_transformers/builders/base.html
[3]: /api_docs/fast_transformers/transformers.html#fast_transformers.transformers.TransformerEncoder
[4]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoder
[5]: /api_docs/fast_transformers/builders/recurrent_encoder_builder.html
