#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement improved clustered causal self attention."""

from math import sqrt

import torch
import torch.autograd
from torch.nn import Dropout, Module
from torch.nn.init import normal_

from ..attention_registry import AttentionRegistry, Optional, Float, Int, \
    Bool, EventDispatcherInstance
from ..events import EventDispatcher
from ..masking import FullMask
from ..aggregate import clustered_aggregate, clustered_broadcast
from ..clustering.hamming import cluster
from ..hashing import compute_hashes
from ..sparse_product import sparse_dot_product, sparse_weighted_average
from ..sparse_product import clustered_sparse_dot_product, \
    clustered_sparse_weighted_average


class _GroupQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, clusters, counts, lengths):
        factors = 1./counts.float()
        q_grouped = clustered_aggregate(Q, clusters, factors, lengths)
        ctx.save_for_backward(clusters, counts, factors)

        return q_grouped

    @staticmethod
    def backward(ctx, grad_q_grouped):
        clusters, counts, factors = ctx.saved_tensors
        grad_q = clustered_broadcast(grad_q_grouped, clusters, counts, factors)

        return grad_q, None, None, None


class _BroadcastValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_grouped, clusters, counts, lengths):
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)
        V = clustered_broadcast(v_grouped, clusters, counts, factors)
        ctx.save_for_backward(clusters, counts, factors, lengths)

        return V

    @staticmethod
    def backward(ctx, grad_v):
        clusters, counts, factors, lengths = ctx.saved_tensors
        grad_v_grouped = clustered_aggregate(grad_v, clusters, factors, lengths)

        return grad_v_grouped, None, None, None, None


class ImprovedClusteredCausalAttention(Module):
    """
    Immproved clustered causal attention approximation by recomputing attention
    for each query with the top-k keys for the corresponding cluster.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to identify the top-k keys with highest
    dot products.

    Subsequently, for each query we compute the sparse dot product with
    the corresponding top-k keys to improve the attention approximation.

    Key difference with improved clustered attention is that we only use
    top-k keys with causal mask, we do not compute attention on the
    bottom-k keys.

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
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, clusters, iterations=10, bits=32,
                 hash_bias=True, topk=32, softmax_temp=None,
                 attention_dropout=0.1, event_dispatcher=""):
        super(ImprovedClusteredCausalAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

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
            query_lengths.lengths.int(),
            clusters=self.clusters,
            iterations=self.iterations,
            bits=self.bits
        )
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _topk_attention(self, Q, K, V,
                        q_flat, q_rev_flat,
                        clusters, counts,
                        topk, topk_values,
                        softmax_temp,
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
            query_lengths.lengths.int()
        )
        # We need to mask out the future
        assert topk.is_contiguous()
        topk_broadcast = clustered_broadcast(
            topk.float(),
            clusters,
            counts,
            torch.ones_like(counts, dtype=torch.float32)
        )
        # Need to be careful here we changed the order of the keys the
        # masking on future needs to be applied in the same way
        seq_ids = torch.arange(L, device=QK.device).view(1, 1, L, 1).repeat(N, H, 1, 1)
        # permute the ids in the same way as input so as to mask the right
        # entries for each query
        s_seq_ids = seq_ids.reshape(-1, 1).index_select(0, q_flat).view(N,H,L,1)
        future_mask = topk_broadcast.long() > s_seq_ids
        QK = QK.masked_fill(
            future_mask,
            float("-1e7")
        )
        A = torch.softmax(softmax_temp * QK, dim=-1)
        # Mask again to ensure no probabilities leak due to float(-1e7)
        # Leakage could be very high as we use a small top-k
        A = A * (1. - future_mask.float())
        A = self.dropout(A)
        assert A.is_contiguous()
        V_new = clustered_sparse_weighted_average(A, V, topk, clusters, counts)

        return V_new

    def _broadcast_values(self, V, clusters, counts, lengths):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), clusters, counts, lengths)
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):

        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(("ImprovedClusteredCausalAttention only supports "
                                "lower triangular masks"))
        queries = queries.permute(0,2,1,3).contiguous()
        keys = keys.permute(0,2,1,3).contiguous()
        values = values.permute(0,2,1,3).contiguous()
        N, H, L, E = queries.shape
        _, _, S, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Cluster the queries into groups
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)
        clusters, counts = groups

        # Re-organize queries so that first group belong to first cluster
        # next to second cluster and so on. This improves kernel implementations.
        # Note that this step is introduced after NeurIPS submission and
        # now the complexity is O(N log(N)).
        q_offset = torch.arange(N*H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N*H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N,H,L,E)

        # Aggregate the re-arranged queries.
        Q_grouped = _GroupQueries.apply(s_queries, *groups, query_lengths.lengths.int())
        # Compute the attention
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        # Set topk to minimum of key lengths if it is smaller than self.topk
        cur_topk = min(self.topk, min(key_lengths.lengths).item())
        topk_values, topk = torch.topk(QK, cur_topk, sorted=False, dim=-1)
        assert topk.is_contiguous()

        # Reverse mapping
        sorted_rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (sorted_rev_indx.view(N*H, -1) + q_offset).reshape(-1)

        # Compute the attention with only the top keys
        V_topk = self._topk_attention(
            s_queries, keys, values,
            q_flat, q_rev_flat,
            clusters, counts,
            topk, topk_values,
            softmax_temp,
            query_lengths
        )
        V_sorted_new = V_topk

        # Reverse the mapping to get correct values
        V_new = V_sorted_new.reshape(-1, D).index_select(0, q_rev_flat).view(N,H,L,D)
        return V_new.permute(0, 2, 1, 3).contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "causal-improved-clustered", ImprovedClusteredCausalAttention,
    [
        ("clusters", Int),
        ("iterations", Optional(Int, 10)),
        ("bits", Optional(Int, 63)),
        ("hash_bias", Optional(Bool, True)),
        ("topk", Optional(Int, 32)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
