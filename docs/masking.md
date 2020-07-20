Masking
=======

In this library, both for convenience and efficiency, we define a [BaseMask][1]
interface that all masks should implement. The BaseMask interface allows
accessing a mask in the following ways:

1. a bool tensor where True signifies what is kept
2. a float tensor where minus infinity signifies what is to be masked
2. a float tensor where zero signifies what is to be masked
3. a length tensor where everything after a certain length is to be masked

This interface allows us to use the same mask definition with various attention
implementations without compromising in performance or requiring code changes.
For instance, softmax masks are usually implemented with additive masks that
contain -inf and linear attention masks are efficiently implemented with
multiplicative masks that contain zeros.

BaseMask
--------

Our [API docs][1] are quite thorough in explaining the BaseMask interface.

Implementations
---------------

We provide three implementations of the BaseMask interface *FullMask*,
*LengthMask* and *TriangularCausalMask*.

### FullMask

```
fast_transformers.masking.FullMask(mask=None, N=None, M=None, device='cpu')
```

The FullMask is a simple wrapper over a pytorch boolean tensor. The arguments
can be given both by keyword arguments and positional arguments. To imitate
function overloading, the constructor checks the type of the first argument and
if it is a tensor it treats it as the mask. otherwise it assumes that it was
the N argument.

**Arguments**

* **mask**: The mask as a PyTorch tensor.
* **N**: The rows of the all True mask to be created if the mask argument is
  not provided.
* **M**: The columns of the all True mask to be created if the mask argument
  is not provided. If N is given M defaults to N.
* **device**: The device to create the mask in (defaults to cpu)

### LengthMask

```
fast_transformers.masking.LengthMask(lengths, max_len=None, device=None)
```

The LengthMask is designed to be used for conveying different lengths of
sequences. It can be accessed as an array of integers which may be beneficial
for some attention implementations.

**Arguments**

* **lengths**: The lengths as a PyTorch long tensor
* **max\_len**: The maximum length for the mask (defaults to lengths.max())
* **device**: The device to be used for creating the masks (defaults to
  lengths.device)

### TriangularCausalMask

```
fast_transformers.masking.TriangularCausalMask(N, device="cpu")
```

Represents a square matrix with everything masked above the main diagonal. It
is meant to be used for training autoregressive transformers.

**Arguments**

* **N**: The size of the matrix
* **device**: The device to create the mask in (defaults to cpu)


[1]: /api_docs/fast_transformers/masking.html#fast_transformers.masking.BaseMask
