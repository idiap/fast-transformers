Attention
==========

The [attention][1] module contains all the implementations of self-attention in
the library. Details for each one are provided in the [API docs][1] but in this
page of the documentation we will mention a few concepts that pertain all the
implementations.

Queries, keys, values
---------------------

Most self-attention implementations project the input queries, keys and values
to multiple heads before computing the new values as a form of weighted average.
Also this weighted average is again passed through a fully connected layer
before returned as the output of the attention module.

All those projections are handled by
[fast_transformers.attention.attention_layer.AttentionLayer][2] which is
described below.

Note, that the AttentionLayer accepts an attention implementation as a first
argument. This allows us to reuse the code that does the query, key, value and
output projections and focus only on implementing efficient attention
mechanisms.

### AttentionLayer

```
fast_transformers.attention.attention_layer.AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)
```

**Arguments**

* **attention**: Specific inner attention implementation that just computes a
  weighted average of values given a similarity of queries and keys.
* **d_model**: The input feature dimensionality
* **n_heads**: The number of heads for the multi head attention
* **d_keys**: The dimensionality of the keys/queries
  (default: d_model/n_heads)
* **d_values**: The dimensionality of the values (default: d_model/n_heads)

Masking
-------

The `forward()` method of all attention implementations accepts the following
three masks, as objects that implement the [BaseMask](masking.md) interface.

* **attn_mask**: This mask encodes the positions of the keys that each query is
  allowed to attend to. It is simply known as the attention mask. In PyTorch it
  is referred to as `attn_mask` or `src_mask`.
* **query_lengths**: This mask, usually a [LengthMask](masking.md#lengthmask),
  encodes the number of queries in each sample of the batch.
* **key_lengths**: Similar to the `query_lengths` mask, this mask encodes the
  number of keys and values in each sample of the batch.

Shapes
------

The [transformer layers](transformers.md) that use the attention modules are
agnostic of the concept of attention heads. They call the attention with
queries, keys and values of the following shape:

Argument | Shape
---------|-----------
queries  | (N, L, D)
keys     | (N, S, D)
values   | (N, S, M)

In the table above, N denotes the batch size, L denotes the maximum number of
queries in a sample, S denotes the maximum number of keys/values in a sample
and D, M are the query/key dimensions and value dimensions respectively.

The [AttentionLayer][2], however, projects the arguments to multiple heads and
calls the attention implementation with the following shapes, where H denotes
the number of heads.

Argument | Shape
---------|-----------
queries  | (N, L, H, D)
keys     | (N, S, H, D)
values   | (N, S, H, M)

&nbsp;

Available Attentions
--------------------

The following is a list with the available attention implementations. Since
this list is not automatically updated we suggest the reader to use the [API
Docs][1] for an exhaustive list of attention implementations.

* [FullAttention][3]
* [LinearAttention][4]
* [CausalLinearAttention][5]
* [ImprovedClusteredAttention][6]
* [ImprovedClusteredCausalAttention][11]
* [ClusteredAttention][7]
* [ReformerAttention][8]
* [LocalAttention][10]
* [ConditionalFullAttention][9]


[1]: /api_docs/fast_transformers/attention/
[2]: /api_docs/fast_transformers/attention/attention_layer.html
[3]: /api_docs/fast_transformers/attention/full_attention.html
[4]: /api_docs/fast_transformers/attention/linear_attention.html
[5]: /api_docs/fast_transformers/attention/causal_linear_attention.html
[6]: /api_docs/fast_transformers/attention/improved_clustered_attention.html
[7]: /api_docs/fast_transformers/attention/clustered_attention.html
[8]: /api_docs/fast_transformers/attention/reformer_attention.html
[9]: /api_docs/fast_transformers/attention/conditional_full_attention.html
[10]: /api_docs/fast_transformers/attention/local_attention.html
[11]: /api_docs/fast_transformers/attention/improved_clustered_causal_attention.html
