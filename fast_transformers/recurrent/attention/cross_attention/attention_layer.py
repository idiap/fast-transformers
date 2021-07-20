#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Similar to the corresponding module in fast_transformers.attention, this
module performs all the query, key, value projections and output projections
leaving the implementation of the attention to the inner attention module.

The crucial difference with respect to the self attention recurrent module
(fast_transformers.recurrent.attention.RecurrentAttentionLayer) is that it
doesn't recompute the projections for the keys and values if the state is not
None.
"""

from torch.nn import Linear, Module

from ....events import EventDispatcher


class RecurrentCrossAttentionLayer(Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer .

    The differences with the aforementioned module as well as the
    RecurrentAttentionLayer are that this module projects the query every time
    and the keys and values only the first time they are provided.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(RecurrentCrossAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, keys, values, key_lengths, state=None):
        """Attend to the keys and values based on the passed in query.

        In the argument description we make use of the following sizes

            - N: the batch size
            - S: the sequence length of the keys and values
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Argument
        --------
            query: (N, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            key_lengths: A fast_transformers.masking.BaseMask implementation
                         that defines the length of each key/value sequence
            state: The state varies depending on the inner attention
                   implementation, but if it is not None then the keys and
                   values are ignored
        """
        #Extract some shapes
        N, _ = query.shape
        H = self.n_heads

        # Project the query
        query = self.query_projection(query).view(N, H, -1)

        # Project the keys and values if there is no state
        if state is None:
            _, S, _ = keys.shape
            keys = self.key_projection(keys).view(N, S, H, -1)
            values = self.value_projection(values).view(N, S, H, -1)
        else:
            keys = None
            values = None

        new_value, state = self.inner_attention(
            query,
            keys,
            values,
            key_lengths,
            state=state
        )
        new_value = new_value.view(N, -1)

        # Project the output and return
        return self.out_projection(new_value), state
