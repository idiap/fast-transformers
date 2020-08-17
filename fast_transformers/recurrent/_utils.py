#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import warnings


def check_state(state=None, memory=None):
    if memory is not None:
        warnings.warn(("'memory' is deprecated for recurrent transformers "
                       " and will be removed in the future, use 'state' "
                       "instead"), DeprecationWarning)
    if state is None:
        state = memory
    return state
