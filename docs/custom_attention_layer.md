Creating a custom attention layer
=================================

In this page, we will go through the process of creating a custom attention
module and integrating it with the library. We will implement a quadratic
kernel attention instead of softmax attention.

New Attention
-------------

Our attention layer will follow closely the implementation of
[FullAttention][1]. Let's start with the skeleton of our module.

```python
class QuadraticAttention(Module):
    def __init__(self, quadratic_temp=1.0, eps=1e-6):
        super(QuadraticAttention, self).__init__()
        self.eps = eps
        self.quadratic_temp = quadratic_temp

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # implement the logic of the layer here
```

The queries, keys and values are already projected and split into multiple
heads by the [AttentionLayer][2]. This means that we need only implement the
attention part.

```python
class QuadraticAttention(Module):
    def __init__(self, quadratic_temp=1.0, eps=1e-6):
        super(QuadraticAttention, self).__init__()
        self.eps = eps
        self.quadratic_temp = quadratic_temp

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # compute the unnormalized attention
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys) # compute the dot products
        QK = torch.square(self.quadratic_temp * QK) # implement our custom attention twist
        QK = QK * attn_mask.float_matrix # use the attention mask as a multiplicative mask
        QK = QK * key_lengths.float_matrix[:, None, None] # also a multiplicative mask

        # normalize and compute the average
        A = QK / (QK.sum(dim=-1, keepdim=True) + self.eps)
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        return V.contiguous()
```

Integrate with the Builder
--------------------------

To add it as an option to the `TransformerEncoderBuilder` or the
`TransformerDecoderBuilder` we have to register our new attention in the
appropriate [attention registry](builders.md#attention-registry). The available
registries are

* AttentionRegistry
* RecurrentAttentionRegistry
* RecurrentCrossAttentionRegistry

Similar to [FullAttention][1] we will use `AttentionRegistry` because our
implementation is not recurrent. The following snippet integrates our quadratic
attention with the builders.

```python
from fast_transformers.attention_registry import AttentionRegistry, \
    Optional, Float  # we also need these to add our new
                     # parameter 'quadratic_temp'

AttentionRegistry.register(
    "square", QuadraticAttention,  # attention_type, class pair
    [
        ("quadratic_temp", Optional(Float, 1.0))  # an optional parameter named
                                                  # 'quadratic_temp' of type
                                                  # float and with default
                                                  # value 1.0
    ]
)
```

Afterwards we can use the builder to create transformers with our new
attention layer.

```python
quadratic_bert = TransformerEncoderBuilder.from_kwargs(
    attention_type="square", # here we select our custom attention layer
    n_layers=12,
    n_heads=12,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=3072,
    activation="gelu",
    quadratic_temp=5.0  # set the temperature for our quadratic layer
)
```


[1]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/full_attention.py
[2]: attention.md
[3]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/builders/attention_builder.py
[4]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/builders/transformer_encoder_builder.py
