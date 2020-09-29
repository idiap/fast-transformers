#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import torch

from .local_product_cpu import local_dot_product as local_dot_product_cpu, \
    local_dot_backward as local_dot_backward_cpu, \
    local_weighted_average as local_weighted_average_cpu, \
    local_weighted_average_backward as local_weighted_average_backward_cpu

try:
    from .local_product_cuda import \
        local_dot_product as local_dot_product_cuda, \
        local_dot_backward as local_dot_backward_cuda, \
        local_weighted_average as local_weighted_average_cuda, \
        local_weighted_average_backward as local_weighted_average_backward_cuda
except ImportError:
    local_dot_product_cuda = None
    local_dot_backward_cuda = None
    local_weighted_average_cuda = None
    local_weighted_average_backward_cuda = None


class LocalDotProduct(torch.autograd.Function):
    """Compute the dot product of the queries and keys but only consider a
    local neighborhood of each query."""
    dot = {
        "cpu": local_dot_product_cpu,
        "cuda": local_dot_product_cuda
    }
    dot_backward = {
        "cpu": local_dot_backward_cpu,
        "cuda": local_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, queries, keys, attn_mask, key_lengths, local_context):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(queries, keys, key_lengths)
        ctx.local_context = local_context

        return LocalDotProduct.dot[queries.device.type](
            queries,
            keys,
            attn_mask,
            key_lengths,
            local_context
        )

    @staticmethod
    def backward(ctx, grad_input):
        queries, keys, key_lengths = ctx.saved_tensors
        local_context = ctx.local_context

        grads = LocalDotProduct.dot_backward[queries.device.type](
            queries,
            keys,
            key_lengths,
            grad_input,
            local_context
        )

        # plus 3 None for masks and local_context
        return grads + (None, None, None)


class LocalWeightedAverage(torch.autograd.Function):
    """Compute the weighted average of the values with the local attention."""
    avg = {
        "cpu": local_weighted_average_cpu,
        "cuda": local_weighted_average_cuda
    }
    avg_backward = {
        "cpu": local_weighted_average_backward_cpu,
        "cuda": local_weighted_average_backward_cuda
    }

    @staticmethod
    def forward(ctx, A, V):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(A, V)

        return LocalWeightedAverage.avg[A.device.type](A, V)

    @staticmethod
    def backward(ctx, grad_input):
        A, V = ctx.saved_tensors
        return LocalWeightedAverage.avg_backward[A.device.type](
            A, V, grad_input
        )


# Alias the autograd functions to python style snake case naming
local_dot_product = LocalDotProduct.apply
local_weighted_average = LocalWeightedAverage.apply
