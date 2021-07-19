#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""This module implements a basic event system that allows the transformer
internal components to make available any tensor with minimal overhead."""

from .event import Event, AttentionEvent, QKVEvent, IntermediateOutput
from .event_dispatcher import EventDispatcher
