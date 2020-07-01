Transformers
============

The [fast\_transformers.transformers](api_docs/fast_transformers/transformers.html)
module provides the `TransformerEncoder` and `TransformerEncoderLayer` classes
that implement a common transformer encoder similar to the PyTorch API.

However, an important difference is that the `TransformerEncoder` **does not
create** the `TransformerEncoderLayer` which allows for injecting a different
implementation with minimal code changes. The encoder layer follows the same
principle and does not create the attention layer but receives it as an
argument which allows for using many different attention implementations with
an otherwise identical model.

We also provide [recurrent transformer encoders](recurrent_transformers.md)
which are meant to be given each input one at a time for autoregressive
inference.

Forward method
--------------

Both the transformer encoder and the transformer encoder layer accept the same
parameters for the `forward()` method.

```
forward(x, attn_mask=None, length_mask=None)
```

**Arguments**

* **x**: The input features of shape (N, L, E) where N is the batch size,
  L is the sequence length (padded) and E is `d_model` passed in the
  constructor.
* **attn_mask**: An implementation of
  [fast_transformers.masking.BaseMask](masking.md) that encodes where each
  element of x can attend to.
* **length_mask**: An implementation of
  [fast_transformers.masking.BaseMask](masking.md) that encodes how many
  elements each sequence in the batch consists of.

If the masks are not provided they are automatically created as an all ones
mask for the attention mask and the size of the tensor for the length mask.

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>Unlike the PyTorch transformer the dimensions of the input are ordered
       with the <b>batch size first and the sequence second</b>.</p>
</div>


TransformerEncoder
------------------

```
fast_transformers.transformers.TransformerEncoder(layers, norm_layer=None)
```

The TransformerEncoder is simply a container for transformer encoder layers
that it receives as a list upon construction. Simply put it is a Sequential
that is aware of [masking](masking.md) and passes the masks to all the
transformer encoder layers.

**Arguments**

* **layers**: A list of TransformerEncoderLayer instances or other nn.Module
  instances that implement the same interface
* **norm\_layer**: A normalization layer to be applied to the final output
  (default: None which means no normalization)

TransformerEncoderLayer
-----------------------

```
fast_transformers.transformers.TransformerEncoderLayer(attention, d_model, n_heads, d_ff=None, dropout=0.1, activation='relu')
```

This transformer encoder layer implements the same encoder layer as PyTorch but
is a bit more open for extension by receiving the attention implementation as a
constructor argument.

**Arguments**

* **attention**: The attention implementation to use given as a nn.Module
* **d\_model**: The input feature dimensionality
* **n\_heads**: The number of heads for the multi head attention
* **d\_ff**: The dimensionality of the intermediate features after the
  attention (default: d\_model*4)
* **dropout**: The dropout rate to apply to the intermediate features
  (default: 0.1)
* **activation**: Choose which activation to use for the feed
  forward part of the layer from the set {'relu', 'gelu'} (default: relu)
