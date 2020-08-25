#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the Reformer attention from the paper
"Reformer the efficient transformer"."""

from math import sqrt

import torch
from torch.nn import Dropout, Module
from torch.nn.init import normal_

from ..attention_registry import AttentionRegistry, Optional, Int, Float, Bool
from ..masking import FullMask


class ReformerAttention(Module):
    """Implement the attention module of the paper "Reformer the efficient
    transformer"

    Arguments
    ---------
        chunk_size  : Chunk size for each block (default: 32)
        bits        : Number of bits for hashing (default: 8)
        rounds      : Number of rounds of attention computation (default: 4)
        masked      : If true, the query does not attend to itsself (default: False)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """

    def __init__(self, chunk_size=32, bits=8, rounds=4, masked=False,
                 softmax_temp=None, attention_dropout=0.1):
        super(ReformerAttention, self).__init__()

        self.chunk_size = chunk_size
        self.bits = bits
        self.rounds = rounds
        self.masked = masked
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)

    def _normalize(self, x):
        norms = torch.sqrt(torch.einsum("nlhe,nlhe->nlh", x, x))
        x_normed = x / norms.unsqueeze(-1)
        return x_normed

    def _look_back(self, x):
        xshape = x.shape

        return torch.cat([
            x.new_zeros((xshape[0], 1) + xshape[2:]),
            torch.repeat_interleave(x, 2, dim=1)[:,:-1]
        ], dim=1).view(xshape[0], xshape[1], 2*xshape[2], *xshape[3:])

    def _reformer_round(self, Q, K, V, mask, softmax_temp):
        # Hash the queries
        N, L, H, E = Q.shape
        planes = Q.new_empty(self.bits, E)
        normal_(planes)
        projected = torch.einsum("nlhe,be->nlhb", K, planes)
        hashes = torch.argmax(
            torch.cat([projected, -projected], dim=-1),
            dim=-1
        )

        # Sort the queries in order to group them
        group = torch.argsort(hashes, dim=1)

        invert_group = torch.empty_like(group)
        batch_indices = torch.arange(N, device=hashes.device).view(N, 1, 1)
        sequence_indices = torch.arange(L, device=hashes.device).view(1, L, 1)
        head_indices = torch.arange(H, device=hashes.device).view(1, 1, H)
        invert_group[batch_indices, group, head_indices] = sequence_indices
        group = group.view(N, -1, self.chunk_size, H)
        invert_group = invert_group.view(N, -1, self.chunk_size, H)
        batch_indices = batch_indices.unsqueeze(1)
        head_indices = head_indices.unsqueeze(0)

        # Reorder Q, V and mask
        Q_grouped = Q[batch_indices, group, head_indices]
        K_grouped = K[batch_indices, group, head_indices]
        V_grouped = V[batch_indices, group, head_indices]
        mask_grouped = mask[
            batch_indices.unsqueeze(1),
            group.unsqueeze(3),
            self._look_back(group).unsqueeze(2)
        ]

        mask_grouped[:, 0, :, :Q_grouped.shape[2]] = float("-inf")

        # When everything is masked just unmask everything because it doesn't
        # matter what the output is at those positions
        # This is to avoid inf/nans in the new values at masked positions
        infmask = torch.isinf(mask_grouped)
        infmask = torch.all(infmask, dim=3, keepdims=True)
        mask_grouped = mask_grouped.masked_fill(infmask, 0.)

        # Attention
        K_grouped = self._look_back(K_grouped)
        QQ = torch.einsum("nblhe,nbshe->nbhls", Q_grouped, K_grouped)
        QQ = QQ + mask_grouped.permute(0, 1, 4, 2, 3)
        A = torch.softmax(softmax_temp * QQ, dim=-1)
        A = self.dropout(A)

        # Values
        V_grouped = self._look_back(V_grouped)
        V_new = torch.einsum("nbhls,nbshe->nblhe", A, V_grouped)
        V_new = V_new.contiguous().view(N, -1,  H, E)
        V_new = V_new[batch_indices, invert_group, head_indices]
        V_new = V_new.contiguous().view(N, L, H, E)
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract the dimensions of query, key, value
        N, L, H, E = queries.shape

        softmax_temp = self.softmax_temp or 1./sqrt(E)
        # Create the mask
        mask = key_lengths.additive_matrix.unsqueeze(1).expand(N, L, L)
        if self.masked:
            mask = mask + torch.eye(L, device=queries.device).unsqueeze(0)*float(-1e9)
       
        if not attn_mask.all_ones:
            mask = mask + attn_mask.additive_matrix.unsqueeze(0)
        # Get normalized Queries as Keys
        K = self._normalize(queries)
        # Zero the masked out keys
        K = K * key_lengths.float_matrix.view(N, L, 1, 1)

        V_new = 0
        factor = 1/self.rounds
        for i in range(self.rounds):
            V_new = V_new + \
                    factor * self._reformer_round(queries, K, values, mask, softmax_temp)

        return V_new


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "reformer", ReformerAttention,
    [
        ("chunk_size", Optional(Int, 32)),
        ("bits", Optional(Int, 32)),
        ("rounds", Optional(Int, 4)),
        ("masked", Optional(Bool, False)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1))
    ]
)
