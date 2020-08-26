Transformers
============

The [fast\_transformers.transformers](/api_docs/fast_transformers/transformers.html)
module provides the `TransformerEncoder` and `TransformerEncoderLayer` classes,
as well as their decoder counterparts, that implement a common transformer
encoder/decoder similar to the PyTorch API.

However, an important difference is that the `TransformerEncoder` **does not
create** the `TransformerEncoderLayer` which allows for injecting a different
implementation with minimal code changes. The encoder layer follows the same
principle and does not create the attention layer but receives it as an
argument which allows for using many different attention implementations with
an otherwise identical model.

We also provide [recurrent transformer encoders and
decoders](recurrent_transformers.md) which are meant to be given each input one
at a time for autoregressive inference.

Forward method
--------------

**TransformerEncoder** or **TransformerEncoderLayer**

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

**TransformerDecoder** or **TransformerDecoderLayer**

```
forward(x, memory, x_mask=None, x_length_mask=None, memory_mask=None, memory_length_mask=None)
```

**Arguments**

* **x**: The input features of shape (N, L, E) where N is the batch size,
  L is the sequence length (padded) and E should be the same as the `d_model`
  passed in the constructor.
* **memory**: The memory features of shape (N, L', E) where N is the
  batch size, L' is the memory's sequence length (padded) and E should be the
  same as the `d_model`.
* **x_mask**: An implementation of fast_transformers.masking.BaseMask
  that encodes where each element of x can attend to in x. Namely the self
  attention mask.
* **x_length_mask**: An implementation of a BaseMask that encodes how
  many elements each sequence in the batch consists of.
* **memory_mask**: An implementation of BaseMask that encodes where each
  element of x can attend to in the memory. Namely the cross attention mask.
* **memory_length_mask**: An implementation of a BaseMask that encodes how
  many elements each memory sequence in the batch consists of.

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
* **n\_heads**: The number of heads for the multi head attention (Note: this
  parameter is unnecessary and will be removed in the near future)
* **d\_ff**: The dimensionality of the intermediate features after the
  attention (default: d\_model*4)
* **dropout**: The dropout rate to apply to the intermediate features
  (default: 0.1)
* **activation**: Choose which activation to use for the feed
  forward part of the layer from the set {'relu', 'gelu'} (default: relu)

TransformerDecoder
------------------

```
fast_transformers.transformers.TransformerDecoder(layers, norm_layer=None)
```

The TransformerDecoder is simply a container for transformer decoder layers.
These layers are passed as a list upon construction. Similar to the
TransformerEncoder, it is a Sequential that is aware of masking and a second
argument `memory` and properly forwards everything to the
TransformerDecoderLayer instances.

**Arguments**

* **layers**: A list of TransformerDecoderLayer instances or other nn.Module
  instances that implement the same interface
* **norm\_layer**: A normalization layer to be applied to the final output
  (default: None which means no normalization)

TransformerDecoderLayer
-----------------------

```
fast_transformers.transformers.TransformerDecoderLayer(self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu')
```

Similar to the encoder layer, this layer implements the decoder that
PyTorch implements but can be used with any attention implementation
because it receives the attention layers as constructor arguments.

* **self\_attention**: The attention implementation to use for self attention
  given as a nn.Module
* **cross\_attention**: The attention implementation to use for cross attention
  given as a nn.Module
* **d\_model**: The input feature dimensionality
* **d\_ff**: The dimensionality of the intermediate features after the
  attention (default: d\_model*4)
* **dropout**: The dropout rate to apply to the intermediate features
  (default: 0.1)
* **activation**: Choose which activation to use for the feed
  forward part of the layer from the set {'relu', 'gelu'} (default: relu)

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <p>The <code>TransformerDecoderLayer</code> accepts different attention
    layers for self attention and cross attention. This allows, for instance,
    for building transformers with linear self attention and softmax cross
    attention.</p>
</div>
