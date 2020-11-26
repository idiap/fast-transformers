Feature Maps
============

The [LinearAttention][1] and [CausalLinearAttention][2] modules, as well as
their corresponding recurrent modules, accept a `feature_map` argument which is
the kernel feature map for each attention implementation. The default
`feature_map` is a simple activation function as used in "[Transformers are
RNNs: Fast Autoregressive Transformers with Linear Attention][3]".

However, the API allows for signficantly more complicated feature maps, that
contain trainable weights or are asymmetric.

FeatureMap API
--------------

All feature maps must implement the following interface.

```python
class FeatureMap(Module):
    def __init__(self, query_dimensions):
        ...

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        ...

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        ...

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        ...

    def forward(self, x):
        # For symmetric feature maps it suffices to define this function
        ...
```

In particular, all feature maps accept the **query dimensions** as the first
constructor parameter. After calling **new_feature_map()** all calls to
**forward** variants should be compatible with each other, namely all
randomness should happen in the **new_feature_map** method. Symmetric feature
maps should only implement **forward**.

Using feature maps
------------------

All modules that accept feature maps, expect a **factory** function. Namely, a
function that when given the query dimensions returns a new feature map
instance.

A simple way to achieve that is by using the `partial()` method of the built-in
module `functools` or the utility class method `factory()` which is basically
the same.

```python
from functools import partial

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.feature_maps import Favor

transformer = TransformerEncoderBuilder.from_kwargs(
    attention_type="linear",
    n_layers=4,
    n_heads=8,
    query_dimensions=32,
    feature_map=Favor.factory(n_dims=256)
).get()

transformer = TransformerEncoderBuilder.from_kwargs(
    attention_type="linear",
    n_layers=4,
    n_heads=8,
    query_dimensions=32,
    feature_map=partial(Favor, n_dims=256)
).get()

```

If you do not want to pass any parameters to the feature map, then it suffices
to use the class object directly.

Available feature maps
----------------------

* [ActivationFunctionFeatureMap][4] uses a simple elementwise activation
  function as a feature map.
* **elu_feature_map** is a specialization of the above where the activation
  function is `elu(x)+1`. It is also the default feature map.
* [RandomFourierFeatures][5] approximates the RBF kernel using random Fourier
  features with trigonometric functions.
* [SmoothedRandomFourierFeatures][8] approximates the RBF kernel plus a
  constant for numerical stability.
* [Favor][6] implements the positive random features designed specifically for
  transformers in the paper "[Rethinking Attention with Performers][7]". It
  should be preferred over the RandomFourierFeatures.
* [GeneralizedRandomFeatures][9] is a simplification of Favor which does not
  approximate softmax but it can increase the rank of the resulting attention
  matrix.


[1]: /api_docs/fast_transformers/attention/linear_attention.html
[2]: /api_docs/fast_transformers/attention/causal_linear_attention.html
[3]: https://arxiv.org/pdf/2006.16236.pdf
[4]: /api_docs/fast_transformers/feature_maps/base.html#fast_transformers.feature_maps.base.ActivationFunctionFeatureMap
[5]: /api_docs/fast_transformers/feature_maps/fourier_features.html#fast_transformers.feature_maps.fourier_features.RandomFourierFeatures
[6]: /api_docs/fast_transformers/feature_maps/fourier_features.html#fast_transformers.feature_maps.fourier_features.Favor
[7]: https://arxiv.org/abs/2009.14794
[8]: /api_docs/fast_transformers/feature_maps/fourier_features.html#fast_transformers.feature_maps.fourier_features.SmoothedRandomFourierFeatures
[9]: /api_docs/fast_transformers/feature_maps/fourier_features.html#fast_transformers.feature_maps.fourier_features.GeneralizedRandomFeatures
