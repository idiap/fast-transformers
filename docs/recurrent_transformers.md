Recurrent Transformers
======================

The transformer layers implemented in the [fast_transformers.transformers][1]
module are processing the entire sequence simultaneously. On the other hand,
this module implements transfomers as recurrent networks. Namely as networks
that process the sequence one element at a time while updating some state.

The TransformerEncoder and TransformerEncoderLayer give way to
[RecurrentTransformerEncoder][2] and [RecurrentTransformerEncoderLayer][3] and
for the decoders [RecurrentTransformerDecoder][7] and
[RecurrentTransformerDecoderLayer][8] respectively.

Forward method
--------------

**RecurrentTransformerEncoder** or **RecurrentTransformerEncoderLayer**

```
forward(x, state=None)
```

**Arguments**

* **x**: The input features of shape (N, E) where N is the batch size and E is
  `d_model` passed in the constructor. Note that x corresponds to a specific
  element in the sequence and not the entire sequence.
* **state**: The state is a python object that varies depending on the
  attention implementation


**RecurrentTransformerDecoder** or **RecurrentTransformerDecoderLayer**

```
forward(x, memory, memory_length_mask=None, state=None)
```

* **x**: The input features of shape (N, E) where N is the batch size and E is
  `d_model` passed in the constructor. Note that x corresponds to a specific
  element in the sequence and not the entire sequence.
* **memory**: A sequence of features (N, S, E) that the input will attend
  to. S is the sequence length and E is the same as for x.
* **memory\_length\_mask**: An implementation of a BaseMask that encodes
  how many elements each memory sequence in the batch consists of.
* **state**: The state is a python object that varies depending on the
  attention implementation

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>The masks are different in the recurrent implementations than in their
    batch counterparts. Namely, recurrent encoders and decoders enforce a
    triangular causal mask on self attention. In addition, recurrent decoders
    enforce a full mask on cross attention.</p>
</div>

Available Attentions
--------------------

Not all attention formulations can be written in an autoregressive fashion as a
recurrent model. In particular, since the sequence is passed to the transformer
element by element we have the same result as passing a causal mask to normal
transformers. The current list for recurrent attention implementations is:

* [LinearAttention][4]
* [FullAttention][5]

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
state = None

x = x0
for i in range(100):
    x, state = model(x, state=state)
```


[1]: /api_docs/fast_transformers/transformers.html
[2]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoder
[3]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerEncoderLayer
[4]: /api_docs/fast_transformers/recurrent/attention/self_attention/linear_attention.html
[5]: /api_docs/fast_transformers/recurrent/attention/self_attention/full_attention.html
[6]: /api_docs/fast_transformers/builders/transformer_builders.html
[7]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerDecoder
[8]: /api_docs/fast_transformers/recurrent/transformers.html#fast_transformers.recurrent.transformers.RecurrentTransformerDecoderLayer
