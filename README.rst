Fast Transformers
=================

Transformers are very successful models that achieve state of the art
performance in many natural language tasks. However, it is very difficult to
scale them to long sequences due to the quadratic scaling of self-attention.

This library was developed for our research on fast attention for transformers.
You can find a list of our papers `in the docs
<https://fast-transformers.github.io>`_ as well as related papers and papers
that we have implemented.

Quick-start
-----------

The following code builds a transformer with softmax attention and one with
linear attention and compares the time required by each to encode a sequence
with 1000 elements.

.. code:: python

    import torch
    from fast_transformers.builders import TransformerEncoderBuilder

    # Create the builder for our transformers
    builder = TransformerEncoderBuilder.from_kwargs(
        n_layers=8,
        n_heads=8,
        query_dimensions=64,
        value_dimensions=64,
        feed_forward_dimensions=1024
    )

    # Build a transformer with softmax attention
    builder.attention_type = "full"
    softmax_model = builder.get()

    # Build a transformer with linear attention
    builder.attention_type = "linear"
    linear_model = builder.get()

    # Construct the dummy input
    X = torch.rand(10, 1000, 8*64)

    # Prepare everythin for CUDA
    X = X.cuda()
    softmax_model.cuda()
    softmax_model.eval()
    linear_model.cuda()
    linear_model.eval()

    # Warmup the GPU
    with torch.no_grad():
        softmax_model(X)
        linear_model(X)
    torch.cuda.synchronize()

    # Measure the execution time
    softmax_start = torch.cuda.Event(enable_timing=True)
    softmax_end = torch.cuda.Event(enable_timing=True)
    linear_start = torch.cuda.Event(enable_timing=True)
    linear_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        softmax_start.record()
        y = softmax_model(X)
        softmax_end.record()
        torch.cuda.synchronize()
        print("Softmax: ", softmax_start.elapsed_time(softmax_end), "ms")
        # Softmax: 144 ms (on a GTX1080Ti)

    with torch.no_grad():
        linear_start.record()
        y = linear_model(X)
        linear_end.record()
        torch.cuda.synchronize()
        print("Linear: ", linear_start.elapsed_time(linear_end), "ms")
        # Linear: 68 ms (on a GTX1080Ti)

Dependencies & Installation
---------------------------

The fast transformers library has the following dependencies:

* PyTorch
* C++ toolchain
* CUDA toolchain (if you want to compile for GPUs)

For most machines installation should be as simple as:

.. code:: bash

    pip install --user pytorch-fast-transformers

Note: macOS users should ensure they have `llvm` and `libomp` installed.
Using the `homebrew <https://brew.sh>`_ package manager, this can be
accomplished by running `brew install llvm libomp`.

Documentation
-------------

There exists a dedicated `documentation site
<https://fast-transformers.github.io/>`_ but you are also encouraged to read
the `source code <https://github.com/idiap/fast-transformers>`_.

Research
--------

Ours
~~~~

To read about the theory behind some attention implementations in this library
we encourage you to follow our research.

* Transformers are RNNs: Fast Autoregressive Transformers with
  Linear Attention (`2006.16236 <https://arxiv.org/abs/2006.16236>`_)
* Fast Transformers with Clustered Attention
  (`2007.04825 <https://arxiv.org/abs/2007.04825>`_)

If you found our research helpful or influential please consider citing

.. code::

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

By others
~~~~~~~~~

* Efficient Attention: Attention with Linear Complexities (`1812.01243
  <https://arxiv.org/abs/1812.01243>`_)
* Linformer: Self-Attention with Linear Complexity (`2006.04768
  <https://arxiv.org/abs/2006.04768>`_)
* Reformer: The Efficient Transformer (`2001.04451
  <https://arxiv.org/abs/2001.04451>`_)

Support, License and Copyright
------------------------------

This software is distributed with the **MIT** license which pretty much means that
you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the
`LICENSE <https://github.com/idiap/fast-transformers/blob/master/LICENSE>`_
file in the repository.
