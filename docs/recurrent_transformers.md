Recurrent Transformers
======================

The transformer layers implemented in the [fast_transformers.transformers][1]
module are processing the entire sequence simultaneously. On the other hand,
this module implements transfomers as recurrent networks. Namely as networks
that process the sequence one element at a time while updating some memory.

Since this API is subject to change we will briefly go over the current
differences in comparison to the normal transformers API.

Forward method
--------------

The TransformerEncoder and TransformerEncoderLayer give way to
[RecurrentTransformerEncoder][2] and [RecurrentTransformerEncoderLayer][3] and
the `forward()` method changes as follows:

```
forward(x, memory=None)
```

**Arguments**

* **x**: The input features of shape (N, E) where N is the batch size and E is
  `d_model` passed in the constructor. Note that x corresponds to a specific
  element in the sequence and not the entire sequence.
* **memory**: The memory is a python object that varies depending on the
  attention implementation

Available Attentions
--------------------

Not all attention formulations can be written in an autoregressive fashion as a
recurrent model. In particular, since the sequence is passed to the transformer
element by element we have the same result as passing a causal mask to normal
transformers. The current list for recurrent attention implementations is:

* [LinearAttention][4]
* [FullAttention][5]

Builder
-------

Building a `RecurrentTransformerEncoder` is very similar to building a
`TransformerEncoder`. We simply provide a different builder named
[RecurrentEncoderBuilder][6] that constructs `RecurrentTransformerEncoder`
models.

Example
-------

The following example builds a random recurrent transformer encoder and applies
its output as input 100 times.

```python
# for simplicity ignore all the classification
# layers and the embedding layers

from fast_transformers.builders import RecurrentEncoderBuilder

model = RecurrentEncoderBuilder.from_kwargs(
    attention_type="linear",
    n_layers=8,
    n_heads=12,
    feed_forward_dimensions=1536,
    query_dimensions=32,
    value_dimensions=32
).get()

x0 = torch.rand(
    10,    # batch size
    12*32  # feature size
)
memory = None

x = x0
for i in range(100):
    x, memory = model(x, memory=memory)
```


[1]: /api_docs/fast_transformers/transformers.html
[2]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoder
[3]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoderLayer
[4]: /api_docs/fast_transformers/recurrent/attention/linear_attention.html
[5]: /api_docs/fast_transformers/recurrent/attention/full_attention.html
[6]: /api_docs/fast_transformers/builders/recurrent_encoder_builder.html
