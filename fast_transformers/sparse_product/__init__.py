#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import torch

from .sparse_product_cpu import \
    sparse_dot_product as sparse_dot_product_cpu, \
    sparse_dot_backward as sparse_dot_backward_cpu, \
    sparse_weighted_average as sparse_weighted_average_cpu, \
    sparse_weighted_average_backward as sparse_weighted_average_backward_cpu
try:
    from .sparse_product_cuda import \
        sparse_dot_product as sparse_dot_product_cuda, \
        sparse_dot_backward as sparse_dot_backward_cuda, \
        sparse_weighted_average as sparse_weighted_average_cuda, \
        sparse_weighted_average_backward as \
            sparse_weighted_average_backward_cuda
except ImportError:
    sparse_dot_product_cuda = None
    sparse_dot_backward_cuda = None
    sparse_weighted_average_cuda = None
    sparse_weighted_average_backward_cuda = None

from .clustered_sparse_product_cpu import \
    clustered_sparse_dot_product as clustered_sparse_dot_product_cpu, \
    clustered_sparse_dot_backward as clustered_sparse_dot_backward_cpu, \
    clustered_sparse_weighted_average as \
        clustered_sparse_weighted_average_cpu, \
    clustered_sparse_weighted_average_backward as \
        clustered_sparse_weighted_average_backward_cpu

try:
    from .clustered_sparse_product_cuda import \
        clustered_sparse_dot_product as clustered_sparse_dot_product_cuda, \
        clustered_sparse_dot_backward as clustered_sparse_dot_backward_cuda, \
        clustered_sparse_weighted_average as \
            clustered_sparse_weighted_average_cuda, \
        clustered_sparse_weighted_average_backward as \
            clustered_sparse_weighted_average_backward_cuda
except ImportError:
    clustered_sparse_dot_product_cuda = None
    clustered_sparse_dot_backward_cuda = None
    clustered_sparse_weighted_average_cuda = None
    clustered_sparse_weighted_average_backward_cuda = None


class SparseDotProduct(torch.autograd.Function):
    """Compute the dot products only at the positions specified by topk."""
    dot = {
        "cpu": sparse_dot_product_cpu,
        "cuda": sparse_dot_product_cuda
    }
    dot_backward = {
        "cpu": sparse_dot_backward_cpu,
        "cuda": sparse_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, topk):
        # Save the inputs to compute the gradient
        ctx.save_for_backward(Q, K, topk)

        # Create the output tensor
        device = Q.device
        N, H, L, E = Q.shape
        _, _, _, k = topk.shape
        product = torch.empty((N, H, L, k), device=device)

        # Actually perform the dot product
        SparseDotProduct.dot[device.type](Q, K, topk, product)

        return product

    @staticmethod
    def backward(ctx, grad_output):
        # Extract the saved tensors and allocate memory for the gradients
        Q, K, topk = ctx.saved_tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)

        SparseDotProduct.dot_backward[Q.device.type](
            Q,
            K,
            topk,
            grad_output,
            grad_Q,
            grad_K
        )

        return grad_Q, grad_K, None


class SparseWeightedAverage(torch.autograd.Function):
    """Compute the weighted average only for the topk values."""
    avg = {
        "cpu": sparse_weighted_average_cpu,
        "cuda": sparse_weighted_average_cuda
    }
    avg_backward = {
        "cpu": sparse_weighted_average_backward_cpu,
        "cuda": sparse_weighted_average_backward_cuda
    }

    @staticmethod
    def forward(ctx, weights, values, topk):
        # Save the tensors to compute the gradient
        ctx.save_for_backward(weights, values, topk)

        # Allocate the output tensor
        N, H, L, _ = weights.shape
        _, _, _, E = values.shape
        output = values.new_zeros(N, H, L, E)

        # Compute the average
        SparseWeightedAverage.avg[weights.device.type](
            weights,
            values,
            topk,
            output
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Extract the saved tensors and allocate memory for the gradients
        weights, values, topk = ctx.saved_tensors
        grad_weights = torch.zeros_like(weights)
        grad_values = torch.zeros_like(values)

        if grad_output.stride()[-1] != 1:
            grad_output = grad_output.contiguous()

        SparseWeightedAverage.avg_backward[weights.device.type](
            weights,
            values,
            topk,
            grad_output,
            grad_weights,
            grad_values
        )

        return grad_weights, grad_values, None


class ClusteredSparseDotProduct(torch.autograd.Function):
    """Compute the dot products only at the positions specified by topk."""
    dot = {
        "cpu": clustered_sparse_dot_product_cpu,
        "cuda": clustered_sparse_dot_product_cuda
    }
    dot_backward = {
        "cpu": clustered_sparse_dot_backward_cpu,
        "cuda": clustered_sparse_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, topk, groups, counts, lengths):
        # Save the inputs to compute the gradient
        ctx.save_for_backward(Q, K, topk, groups)

        device = Q.device
        N, H, L, E = Q.shape
        _, _, C, k = topk.shape

        # Create the output tensor
        product = torch.empty((N, H, L, k), device=device)

        # Unfortunately the cpu and gpu interfaces are different so
        # the entire call is surrounded by if-else block
        if device.type == "cpu":
            ClusteredSparseDotProduct.dot[device.type](
                Q,
                K,
                groups,
                topk,
                product
            )

        else:
            queries_per_block = min(L, 1024//k) 
            threads = k * queries_per_block
            blocks = ((L*k)//threads) + C + 1
            query_map = torch.ones((N, H, blocks), dtype=torch.int32).cuda() * L 
            blocks_map = torch.ones((N, H, blocks), dtype=torch.int32).cuda() * -1 
            _, sorted_group_indices = torch.sort(groups, descending=True, dim=-1)

            # Actually perform the dot product
            ClusteredSparseDotProduct.dot[device.type](
                Q,
                K,
                topk,
                lengths,
                blocks_map,
                query_map,
                counts,
                sorted_group_indices,
                product
            )
        return product

    @staticmethod
    def backward(ctx, grad_output):
        # Extract the saved tensors and allocate memory for the gradients
        Q, K, topk, groups = ctx.saved_tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        ClusteredSparseDotProduct.dot_backward[Q.device.type](
            Q,
            K,
            groups,
            topk,
            grad_output,
            grad_Q,
            grad_K
        )

        return grad_Q, grad_K, None, None, None, None

class ClusteredSparseWeightedAverage(torch.autograd.Function):
    """Compute the weighted average only for the topk values."""
    avg = {
        "cpu": clustered_sparse_weighted_average_cpu,
        "cuda": clustered_sparse_weighted_average_cuda
    }
    avg_backward = {
        "cpu": clustered_sparse_weighted_average_backward_cpu,
        "cuda": clustered_sparse_weighted_average_backward_cuda
    }

    @staticmethod
    def forward(ctx, weights, values, topk, groups):
        # Save the tensors to compute the gradient
        ctx.save_for_backward(weights, values, topk, groups)

        # Allocate the output tensor
        N, H, L, _ = weights.shape
        _, _, _, E = values.shape
        output = values.new_zeros(N, H, L, E)
        
        # Compute the average
        ClusteredSparseWeightedAverage.avg[weights.device.type](
            weights,
            values,
            groups,
            topk,
            output
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Extract the saved tensors and allocate memory for the gradients
        weights, values, topk, groups = ctx.saved_tensors
        grad_weights = torch.zeros_like(weights)
        grad_values = torch.zeros_like(values)

        if grad_output.stride()[-1] != 1:
            grad_output = grad_output.contiguous()

        ClusteredSparseWeightedAverage.avg_backward[weights.device.type](
            weights,
            values,
            groups,
            topk,
            grad_output,
            grad_weights,
            grad_values
        )

        return grad_weights, grad_values, None, None


# Alias the autograd functions to python style snake case naming
clustered_sparse_dot_product = ClusteredSparseDotProduct.apply
clustered_sparse_weighted_average = ClusteredSparseWeightedAverage.apply

# Alias the autograd functions to python style snake case naming
sparse_dot_product = SparseDotProduct.apply
sparse_weighted_average = SparseWeightedAverage.apply
