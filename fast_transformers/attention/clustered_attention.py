#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement clustered self attention."""

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


class ClusteredAttention(Module):
    """Use LSH and clustering in the resulting Hamming space to group queries
    that will have minimal L2 distance from each other.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to compute the attention using:
    
        V'_c = softmax(Q_c.mm(K.t()), dim=-1).mm(V).

    Now the computed values V'_c are "broadcasted" back to the query members
    of the corresponding cluster.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, clusters, iterations=10, bits=32,
                 hash_bias=True, softmax_temp=None, attention_dropout=0.1):
        super(ClusteredAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
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
        groups =  cluster(
            hashes,
            query_lengths._lengths.int(),
            clusters=self.clusters,
            iterations=self.iterations,
            bits=self.bits
        )
        return groups

    def _group_queries(self, Q, groups):
        """Aggregate the Qs based on the index of cluster they belong to. Make
        sure to allow for gradient propagation backwards from the grouped
        queries to each query."""
        q_grouped = _GroupQueries.apply(Q, *groups)
        return q_grouped

    def _broadcast_values(self, V, groups):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), *groups)
        V_new = V_new.permute(0, 2, 1, 3).contiguous()
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Make sure that there is no attention mask
        assert attn_mask.all_ones, ("Clustered attention cannot use an "
                                    "arbitrary attention mask.")

        queries = queries.permute(0,2,1,3).contiguous()
        keys = keys.permute(0,2,1,3).contiguous()
        values = values.permute(0,2,1,3).contiguous()

        N, H, L, E = queries.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)
        
        # Cluster the queries into groups
        groups = self._create_query_groups(queries, query_lengths)
        Q_grouped = self._group_queries(queries, groups)
        # Compute the attention
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nhsd->nhld", A, values)

        # Broadcast grouped attention
        return self._broadcast_values(V, groups)


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "clustered", ClusteredAttention,
    [
        ("clusters", Int),
        ("iterations", Optional(Int, 10)),
        ("bits", Optional(Int, 32)),
        ("hash_bias", Optional(Bool, True)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1))
    ]
)
