#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement the positive orthogonal random features from the paper
"Rethinking Attention with Performers" https://arxiv.org/pdf/2009.14794.pdf
and the traditional random Fourier features that approximate the RBF kernel.
"""

from math import sqrt, log
import warnings

import torch

from .base import FeatureMap


def orthogonal_random_matrix_(w):
    """Initialize the matrix w in-place to compute orthogonal random features.

    The matrix is initialized such that its columns are orthogonal to each
    other (in groups of size `rows`) and their norms is drawn from the
    chi-square distribution with `rows` degrees of freedom (namely the norm of
    a `rows`-dimensional vector distributed as N(0, I)).

    Arguments
    ---------
        w: float tensor of size (rows, columns)
    """
    rows, columns = w.shape
    start = 0
    while start < columns:
        end = min(start+rows, columns)
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.linalg.qr(block)
        w[:, start:end] = (
            Q[:, :end-start] * norms[None, :end-start]
        )
        start += rows


class RandomFourierFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].

    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, redraw=1, deterministic_eval=False):
        super(RandomFourierFeatures, self).__init__(query_dimensions)

        self.n_dims = n_dims or query_dimensions
        self.query_dimensions = query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = (
            1/sqrt(query_dimensions) if softmax_temp is None
            else softmax_temp
        )
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(self.query_dimensions, self.n_dims//2)
        )
        self._calls = -1

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.query_dimensions,
            self.n_dims//2,
            device=device
        )
        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()
        self.register_buffer("omega", omega)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)


class SmoothedRandomFourierFeatures(RandomFourierFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.

    Implements K(x, y) = exp(-|x-y|^2) + s.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        smoothing: float, The smoothing parameter to add to the dot product.
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, smoothing=1.0, redraw=1,
                 deterministic_eval=False):
        super(SmoothedRandomFourierFeatures, self).__init__(
            query_dimensions,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal,
            redraw=redraw,
            deterministic_eval=deterministic_eval
        )
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(
            y.shape[:-1] + (1,),
            self.smoothing,
            dtype=y.dtype,
            device=y.device
        )
        return torch.cat([y, smoothing], dim=-1)


class Favor(RandomFourierFeatures):
    """Positive orthogonal random features that approximate the softmax kernel.

    Basically implementation of Lemma 1 from "Rethinking Attention with
    Performers".

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=True, stabilize=False, redraw=1,
                 deterministic_eval=False):
        super(Favor, self).__init__(query_dimensions, n_dims=n_dims,
                                    softmax_temp=softmax_temp,
                                    orthogonal=orthogonal, redraw=redraw,
                                    deterministic_eval=deterministic_eval)
        self.stabilize = stabilize

    def _check_sequence_length(self, x):
        """Check that the 2nd dimension is larger than the 3rd as a heuristic
        that the sequence length will be larger than the number of heads. If
        not simply warn of a possible bug."""
        if len(x.shape) != 4:
            warnings.warn(("Favor.stabilize is set to True but the input "
                           "feature does not have the shape (N, L, H, D) "
                           "which may result in unexpected behaviour"))

        if x.shape[1] < x.shape[2]:
            warnings.warn(("Favor.stabilize is set to True but the 2nd "
                           "dimension of the input is smaller than the 3rd "
                           "which could indicate that the sequence length and "
                           "the heads are flipped. This may result in incorrect "
                           "behaviour. The shape of the input is "
                           "{!r}.").format(x.shape))

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared/2)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared * 0.5 + 0.5 * log(self.n_dims)

        # If stabilize is True then add the max norm per sequence in order to
        # ensure that exp_u1 and exp_u2 will be <1.
        #
        # NOTE: This is the only part of this feature map that assumes the
        #       2nd dimension is the sequence length. We call the
        #       _check_sequence_length dimension function to be able to catch
        #       some possible bugs ahead of time.
        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi


class GeneralizedRandomFeatures(RandomFourierFeatures):
    """Implements the generalized random Fourier features from Performers.

    It computes φ(χ) = [f(ω_1 χ), f(ω_2 χ), ..., f(ω_n χ)] where f(.) is the
    passed in `kernel_fn`.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (default: query_dimensions)
        softmax_temp: float, A normalizer for the dot products that is
                     multiplied to the input features before the feature map
                     application (default: 1.0)
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        kernel_fn: callable, defines the f used for the feature map.
                   (default: relu)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=1.0,
                 orthogonal=True, kernel_fn=torch.relu, redraw=1,
                 deterministic_eval=False):
        super(GeneralizedRandomFeatures, self).__init__(
            query_dimensions,
            n_dims=2*query_dimensions if n_dims is None else 2*n_dims,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal,
            redraw=redraw,
            deterministic_eval=deterministic_eval
        )
        self.kernel_fn = kernel_fn

    def forward(self, x):
        if self.softmax_temp != 1.0:
            x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        return self.kernel_fn(u)
