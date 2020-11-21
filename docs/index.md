Fast Transformers
=================

Transformers are very succsessfull models that achieve state of the art
performance in many natural language tasks. However, it is very difficult to
scale them to long sequences due to the quadratic scaling of self-attention.

This library was developed for our research on fast attention for transformers.
You can find a list of our papers [below](#research) as well as related papers
and papers that we have implemented.

Quick-start
-----------

The main interface of the library for using the implemented fast transformers
is the [builder interface](api/fast_transformers/builders/). This allows for
experimenting with different attention implentations with minimal code changes.
For instance building a BERT-like transformer encoder is as simple as the
following code:

```python
import torch
from fast_transformers.builders import TransformerEncoderBuilder

# Build a transformer encoder
bert = TransformerEncoderBuilder.from_kwargs(
    n_layers=12,
    n_heads=12,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=3072,
    attention_type="full", # change this to use another
                           # attention implementation
    activation="gelu"
).get()

y = bert(torch.rand(
    10,    # batch_size
    512,   # sequence length
    64*12  # features
))
```

Installation
------------

The fast transformers library has the following dependencies:

* PyTorch
* C++ toolchain
* CUDA toolchain (if you want to compile for GPUs)

For most machines installation should be as simple as:

```bash
pip install --user pytorch-fast-transformers
```

Research
--------

### Ours

To read about the theory behind some attention implementations in this library
we encourage you to follow our research.

* Transformers are RNNs: Fast Autoregressive Transformers with
  Linear Attention ([arxiv](https://arxiv.org/abs/2006.16236),
  [video](https://youtu.be/KBWh7XCUAi8))
* Fast Transformers with Clustered Attention
  ([arxiv](https://arxiv.org/abs/2007.04825),
  [blog](https://clustered-transformers.github.io/blog/))

If you found our research helpful or influential please consider citing

```
@inproceedings{katharopoulos_et_al_2020,
    author = {Katharopoulos, A. and Vyas, A. and Pappas, N. and Fleuret, F.},
    title = {Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year = {2020}
}

@article{vyas_et_al_2020,
    author={Vyas, A. and Katharopoulos, A. and Fleuret, F.},
    title={Fast Transformers with Clustered Attention},
    booktitle = {Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)},
    year={2020}
}
```

### By others

* Efficient Attention: Attention with Linear Complexities ([arxiv](https://arxiv.org/abs/1812.01243))
* Linformer: Self-Attention with Linear Complexity ([arxiv](https://arxiv.org/abs/2006.04768))
* Reformer: The Efficient Transformer ([arxiv](https://arxiv.org/abs/2001.04451))

Support, License and Copyright
------------------------------

This software is distributed with the **MIT** license which pretty much means that
you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the
[LICENSE](https://github.com/idiap/fast-transformers/blob/master/LICENSE) file
in the repository.
