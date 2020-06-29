#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""The weight mapper module provides a utility to load transformer model
weights from other implementations to a fast_transformers model.

NOTE: This API is lkely to change in the future as we collect more information
      regarding how people use it.
"""

import re


class MappingRule(object):
    """A mapping rule can be applied to a key and value and it returns new keys
    and values to be added in the state dict."""
    def matches(self, key):
        """Check whether this mapping rule should be applied to this key."""
        raise NotImplementedError()

    def apply(self, key, value):
        """Apply the rule and map the key to a new one."""
        raise NotImplementedError()


class IdentityRule(MappingRule):
    """The identity rule matches all keys and returns them as is."""
    def matches(self, key):
        return True

    def apply(self, key, value):
        return [(key, value)]


class NotRule(MappingRule):
    """Decorate a MappingRule by using a logical not for the matches function
    and identity for the apply."""
    def __init__(self, rule):
        self.rule = rule

    def matches(self, key):
        return not self.rule.matches(key)

    def apply(self, key, value):
        return [(key, value)]

class OrRule(MappingRule):
    """Decorate some MappingRules using the logical or to create a matches
    function that returns True if any of the rules matches. In case of a match
    apply all of the rules."""
    def __init__(self, *rules):
        self.rules = rules

    def matches(self, key):
        return any(r.matches(key) for r in self.rules)

    def apply(self, key, value):
        items = [(key, value)]
        for r in self.rules:
            items = [
                r.apply(k, v)
                for k, v in items
            ]
        return items


class RegexRule(MappingRule):
    """Apply a regex search and replace on a key.

    Arguments
    ---------
        search: str, the regex pattern to search and replace
        replace: str or callable, the replacement for every occurence of the
                 search pattern. If it is a callable it should follow the rules
                 of python re.sub().
    """
    def __init__(self, search, replace):
        self.pattern = re.compile(search)
        self.replace = replace

    def matches(self, key):
        return self.pattern.search(key) is not None

    def apply(self, key, value):
        return [(self.pattern.sub(self.replace, key), value)]


class PytorchAttentionWeightsRule(MappingRule):
    """Map the merged MultiheadAttention weights to the corresponding keys and
    values."""
    def __init__(self):
        self.weight_pattern = "self_attn.in_proj_weight"
        self.bias_pattern = "self_attn.in_proj_bias"

    def matches(self, key):
        return (
            self.weight_pattern in key or
            self.bias_pattern in key
        )

    def apply(self, key, value):
        N = value.shape[0]
        if self.weight_pattern in key:
            return [
                (
                    key.replace(
                        self.weight_pattern,
                        "attention.query_projection.weight"
                    ),
                    value[:N//3]
                ),
                (
                    key.replace(
                        self.weight_pattern,
                        "attention.key_projection.weight"
                    ),
                    value[N//3:2*N//3]
                ),
                (
                    key.replace(
                        self.weight_pattern,
                        "attention.value_projection.weight"
                    ),
                    value[2*N//3:]
                )
            ]
        if self.bias_pattern in key:
                return [
                    (
                        key.replace(
                            self.bias_pattern,
                            "attention.query_projection.bias"
                        ),
                        value[:N//3]
                    ),
                    (
                        key.replace(
                            self.bias_pattern,
                            "attention.key_projection.bias"
                        ),
                        value[N//3:2*N//3]
                    ),
                    (
                        key.replace(
                            self.bias_pattern,
                            "attention.value_projection.bias"
                        ),
                        value[2*N//3:]
                    )
                ]


class SimpleMapper(object):
    """Map keys of a state dict to other keys.

    Arguments
    ---------
        rules: A list of mapping rules to apply to the keys (default: []).
        add_identity: bool, if set to True add a catch all identity rule as the
                      final rule (default: True).
    """
    def __init__(self, rules=[], add_identity=True):
        self._rules = rules
        if add_identity:
            self._rules.append(IdentityRule())

    def map(self, state_dict):
        new_state = {}
        for k, v in state_dict.items():
            for rule in self._rules:
                if rule.matches(k):
                    for nk, nv in rule.apply(k, v):
                        new_state[nk] = nv
                    break
        return new_state

    @classmethod
    def load_file(cls, filepath, model_root=None, map_location=None,
                  **other_args):
        """Load the file and apply the weight map.

        The model root the key that contains the state dict to be mapped.

        Arguments
        ---------
            filepath: The file containing the saved state.
            model_root: The key for the state dict to be mapped, if None assume
                        it is the top level dictionary (default: None).
            map_location: The parameter is passed to torch.load .
            other_args: The parameter dict is passed to torch.load because it
                        expects a similar dictionary of arguments to pass to
                        pickle.load.
        """
        state = torch.load(filepath, map_location=map_location, **other_args)
        if model_root is None:
            state = cls().map(state)
        else:
            state[model_root] = cls().map(state[model_root])

        return state


class PytorchMapper(SimpleMapper):
    """Map a Pytorch transformer encoder state dict to a fast transformers
    one."""
    def __init__(self):
        super(PytorchMapper, self).__init__([
            PytorchAttentionWeightsRule(),
            RegexRule(
                r"layers\.(\d+)\.self_attn\.([a-z]+)_proj(ection)?\.",
                r"layers.\1.attention.\2_projection."
            ),
            NotRule(OrRule(
                RegexRule(
                    r"\.softmax_temp$",
                    r""
                )
            ))
        ], add_identity=False)


class HugginfaceBertEncoderMapper(SimpleMapper):
    """Map the weights of a model that uses a BertEncoder to our fast
    transformers."""
    RULES = [
        RegexRule(
            r"layer\.(\d+)\.attention\.self\.(query|key|value)",
            r"layers.\1.attention.\2_projection"
        ),
        RegexRule(
            r"layer\.(\d+)\.attention\.output\.dense",
            r"layers.\1.attention.out_projection"
        ),
        RegexRule(
            r"layer\.(\d+)\.attention\.output\.LayerNorm",
            r"layers.\1.norm1"
        ),
        RegexRule(
            r"layer\.(\d+)\.intermediate\.dense",
            r"layers.\1.linear1"
        ),
        RegexRule(
            r"layer\.(\d+)\.output\.dense",
            r"layers.\1.linear2"
        ),
        RegexRule(
            r"layer\.(\d+)\.output\.LayerNorm",
            r"layers.\1.norm2"
        )
    ]

    def __init__(self):
        super(HugginfaceBertEncoderMapper, self).__init__(self.RULES)


class LongformerMapper(SimpleMapper):
    """Map the longformer weights to our fast transformers.

    NOTE: The projections for the global attention are ignored.
    """
    def __init__(self):
        super(LongformerMapper, self).__init__(
            HugginfaceBertEncoderMapper.RULES + [
                NotRule(RegexRule(
                    r"layer\.(\d+)\.attention\.self\.(query|key|value)_global",
                    ""
                ))
            ],
            add_identity=False
        )
