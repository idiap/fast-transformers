Tips & Tricks
=============

In this module we will provide examples of common usecases when using the fast
transformers library. We will be adding more examples as more utilities are
implemented.

Mirrored networks
---------------

We call mirrored networks, networks that _share the parameter instances_ but have
different module implementations. The most common use case is to have mirrored
batch and recurrent versions of the same transformer model in order to train
with the batch version and evaluate using the recurrent version.

We provide the utility `make_mirror(src_module, dst_module)` to automatically
set the source module parameters to the destination module.

```python
from fast_transformer.builders import TransformerEncoderBuilder, \
    RecurrentEncoderBuilder
from fast_transfomer.utils import make_mirror

params = dict(...)
transformer = TransformerEncoderBuilder.from_dictionary(params).get()
recurrent_transformer = RecurrentEncoderBuilder.from_dictionary(params).get()
make_mirror(transformer, recurrent_transformer)

# Now training transformer also changes the parameters of recurrent transformer
# and vice-versa.
```

Checkpointing
---------------

[Checkpointing](https://pytorch.org/docs/stable/checkpoint.html) is important
when training large neural networks to allow for more layers to fit in a single
GPU. The default PyTorch method of checkpointing, only accepts tensors as
arguments which unfortunately excludes our self-attention and transformer
modules that expect `BaseMask` objects for masking.

!!! tip "Under development"
    We are developing wrappers around the default checkpointing mechanisms that
    will allow users to checkpoint modules of their choosing or even checkpoint
    every transformer block in a transformer encoder or decoder.

    Check back for details or check our [github repository issue #21][1].

[1]: https://github.com/idiap/fast-transformers/issues/21
