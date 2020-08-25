#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement improved clustered self attention."""

from math import sqrt

import torch
import torch.autograd
from torch.nn import Dropout, Module
from torch.nn.init import normal_

from ..attention_registry import AttentionRegistry, Optional, Float, Int, Bool
from ..masking import FullMask
from ..aggregate import aggregate, broadcast
from ..clustering.hamming import cluster
from ..hashing import compute_hashes
from ..sparse_product import sparse_dot_product, sparse_weighted_average
from ..sparse_product import clustered_sparse_dot_product, \
    clustered_sparse_weighted_average


class _GroupQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, clusters, counts):
        factors = 1/counts.float()
        q_grouped = aggregate(Q, clusters, factors)
        ctx.save_for_backward(clusters, factors)

        return q_grouped

    @staticmethod
    def backward(ctx, grad_q_grouped):
        clusters, factors = ctx.saved_tensors
        grad_q = broadcast(grad_q_grouped, clusters, factors)

        return grad_q, None, None


class _BroadcastValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_grouped, clusters, counts):
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)
        V = broadcast(v_grouped, clusters, factors)
        ctx.save_for_backward(clusters, factors)

        return V

    @staticmethod
    def backward(ctx, grad_v):
        clusters, factors = ctx.saved_tensors
        grad_v_grouped = aggregate(grad_v, clusters, factors)

        return grad_v_grouped, None, None


class ImprovedClusteredAttention(Module):
    """
    Immproved clustered attention approximation by recompution attention
    for each query with the top-k keys for the corresponding cluster.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to identify the top-k keys with highest
    dot products.
    
    Subsequently, for each query we compute the sparse dot product with
    the corresponding top-k keys to improve the attention approximation.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        topk: Number of top-k keys to for improved approximation (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, clusters, iterations=10, bits=32,
                 hash_bias=True, topk=32, softmax_temp=None, attention_dropout=0.1):
        super(ImprovedClusteredAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)

    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape

        # Compute the hashes for all the queries
        planes = Q.new_empty((self.bits, E+1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)

        # Cluster the hashes and return the cluster index per query
        clusters, counts =  cluster(
            hashes,
            query_lengths._lengths.int(),
            clusters=self.clusters,
            iterations=self.iterations,
            bits=self.bits
        )
        return clusters, counts

    def _topk_attention(self, Q, K, V,
                        clusters, counts,
                        topk, topk_values,
                        A_bottomk, softmax_temp,
                        query_lengths):
        """Return the attention with just the topk heads."""
        # Extract some indices
        N, H, L, E = Q.shape
        _, _, S, _ = K.shape
        _, _, C, k = topk.shape

        # We need to pass the output tensor to initialize to 0
        QK = clustered_sparse_dot_product(
            Q, K, topk,
            clusters, counts,
            query_lengths._lengths.int()
        )
        # We need to mask the topk dot products if topk > input_length
        QK = QK.masked_fill(
            torch.isinf(topk_values[:,0,0,:]).view(N, 1, 1, k),
            float("-inf")
        )
        A = torch.softmax(softmax_temp * QK, dim=-1)
        assert A_bottomk.is_contiguous()
        A_bottomk = broadcast(
            A_bottomk.unsqueeze(3),
            clusters,
            torch.ones_like(counts, dtype=torch.float32)
        )
        A = A * (1.0 - A_bottomk)
        A = self.dropout(A)
        assert A.is_contiguous()
        V_new = clustered_sparse_weighted_average(A, V, topk, clusters)

        return V_new

    def _broadcast_values(self, V, clusters, counts):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), clusters, counts)
        return V_new

    def _bottomk_attention(self, QK, V, clusters, counts, topk, softmax_temp): 
        """Return the attention with just the bottomk keys."""
        N, H, C, S = QK.shape

        A = torch.softmax(softmax_temp * QK, dim=-1)
        mask = QK.new_ones(QK.shape)
        mask[
            torch.arange(N, device=QK.device).view(N, 1, 1, 1),
            torch.arange(H, device=QK.device).view(1, H, 1, 1),
            torch.arange(C, device=QK.device).view(1, 1, C, 1),
            topk,
        ] = 0
        A = A * mask
        A_bottomk = A.sum(-1)
        A = self.dropout(A)
        # Compute the values
        V_new = torch.einsum("nhls,nhse->nhle", A, V)
        # Broadcast the values back depending on the groups
        V_new = self._broadcast_values(V_new, clusters, counts)
        
        return V_new, A_bottomk

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Make sure that there is no attention mask
        assert attn_mask.all_ones, ("Improved-clustered attention cannot "
                                    "use an arbitrary attention mask.")

        queries = queries.permute(0,2,1,3).contiguous()
        keys = keys.permute(0,2,1,3).contiguous()
        values = values.permute(0,2,1,3).contiguous()
        N, H, L, E = queries.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)
        
        # Cluster the queries into groups
        clusters, counts = self._create_query_groups(queries, query_lengths)
        Q_grouped = _GroupQueries.apply(queries, clusters, counts)
        # Compute the attention
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        topk_values, topk = torch.topk(QK, self.topk, sorted=False, dim=-1)
        assert topk.is_contiguous()

        # Now compute the attention with only the bottom keys
        V_bottomk, A_bottomk = self._bottomk_attention(
            QK, values,
            clusters, counts,
            topk,
            softmax_temp
        )

        # Now compute the attention with only the top keys
        V_topk = self._topk_attention(
            queries, keys, values,
            clusters, counts,
            topk, topk_values,
            A_bottomk,
            softmax_temp,
            query_lengths
        )
        V_new = V_topk + V_bottomk

        return V_new.permute(0, 2, 1, 3).contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "improved-clustered", ImprovedClusteredAttention,
    [
        ("clusters", Int),
        ("iterations", Optional(Int, 10)),
        ("bits", Optional(Int, 32)),
        ("hash_bias", Optional(Bool, True)),
        ("topk", Optional(Int, 32)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1))
    ]
)
