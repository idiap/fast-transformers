#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement the positive orthogonal random features from the paper
"Rethinking Attention with Performers" https://arxiv.org/pdf/2009.14794.pdf
"""

from math import sqrt, log

import torch

from .base import FeatureMap


def orthogonal_random_matrix_(w):
    rows, columns = w.shape
    start = 0
    while start < columns:
        end = min(start+rows, columns)
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.qr(block)
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
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False):
        super(RandomFourierFeatures, self).__init__(query_dimensions)

        self.n_dims = n_dims or query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = (
            1/sqrt(query_dimensions) if softmax_temp is None
            else softmax_temp
        )

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(query_dimensions, self.n_dims//2)
        )

    def new_feature_map(self):
        if self.orthogonal:
            orthogonal_random_matrix_(self.omega)
        else:
            self.omega.normal_()

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)


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
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=True, stabilize=False):
        super(Favor, self).__init__(query_dimensions, n_dims=n_dims,
                                    softmax_temp=softmax_temp,
                                    orthogonal=orthogonal)
        self.stabilize = stabilize

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared/2)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared * 0.5 + 0.5 * log(self.n_dims)
        if self.stabilize:
            offset = offset + norm_x_squared.max()

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi
