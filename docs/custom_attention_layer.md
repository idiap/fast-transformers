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
    def __init__(self, eps=1e-6):
        super(QuadraticAttention, self).__init__()
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # implement the logic of the layer here
```

The queries, keys and values are already projected and split into multiple
heads by the [AttentionLayer][2]. This means that we need only implement the
attention part.

```python
class QuadraticAttention(Module):
    def __init__(self, eps=1e-6):
        super(QuadraticAttention, self).__init__()
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # compute the unnormalized attention
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys) # compute the dot products
        QK = torch.square(QK) # implement our custom attention twist
        QK = QK * attn_mask.float_matrix # use the attention mask as a multiplicative mask
        QK = QK * key_lengths.float_matrix[:, None, None] # also a multiplicative mask

        # normalize and compute the average
        A = QK / (QK.sum(dim=-1, keepdim=True) + self.eps)
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        return V.contiguous()
```

Integrate with the Builder
--------------------------

To add it as an option to the `TransformerEncoderBuilder` we have to edit two
files. Firstly, we have to add it as an option to the
[AttentionBuilder.attention_type][3] as follows:

```python
class AttentionBuilder(object):
    ...
    ...
    @attention_type.setter
    def attention_type(self, val):
        attentions = ["full", "clustered", "improved-clustered",
                      "improved-causal", "linear", "causal-linear",
                      "reformer", "exact-topk", "square"] # add the 'square' to the list
        if val not in attentions:
            raise ValueError(("{!r} is not one of the available attention "
                              "types {!r}").format(val, attentions))
        self._attention_type = val
    ...
    ...
```

Secondly, we have to edit the [TransformerEncoderBuilder][4] to create the
correct attention layer when the attention\_type is set to 'square'.

```python
class TransformerEncoderBuilder(BaseTransformerBuilder):
    ...
    ...
    def _get_attention(self):
        attentions = {
            ...
            ...
            "square": QuadraticAttention
        }
        ...
        ...
```

After those changes we can use the builder to create transformers with our new
attention layer.

```python
quadratic_bert = TransformerEncoderBuilder.from_kwargs(
    attention_type="square", # here we select our custom attention layer
    n_layers=12,
    n_heads=12,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=3072,
    activation="gelu"
)
```

Future changes
--------------

Editing two files to add a new attention implementation to a transformer
builder is obviously suboptimal. We are in the process of changing this
procedure with a more streamlined way of registering new attention layers to
the transformer builders.


[1]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/full_attention.py
[2]: attention.md
[3]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/builders/attention_builder.py
[4]: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/builders/transformer_encoder_builder.py
