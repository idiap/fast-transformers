Events
======

When training transformers, some internal representations, such as the
attention matrices, are useful for identifying problems or understanding how
the model works.

Instead of making these representations accessible by returning them as the
output of the model, we provide them via an event system. This allows for
greater flexibility by allowing different attention implementations to return
different things without affecting the execution speed or the interfaces.

You can explore the interfaces of the event system in our [API Docs][2].

Getting Started
---------------

Before delving deeper into the API of the event system, the following commented
code snippet collects all the attention matrices from a forward pass of a
transformer and plots the first head of the first sample using matplotlib.

```python
import matplotlib.pyplot as plt
import torch

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.events import EventDispatcher, AttentionEvent

# Make a transformer as we always would
transformer = TransformerEncoderBuilder.from_kwargs(
    n_layers=4,
    n_heads=4,
    query_dimensions=64,
    value_dimensions=64
).get()

# Make an event handler that just appends to a list
attentions = []
def save_attention_matrix(event):
    attentions.append(event.attention_matrix.detach().cpu())

# Register said event handler for AttentionEvents
EventDispatcher.get().listen(AttentionEvent, save_attention_matrix)

# Do a forward pass like always
transformer(torch.rand(10, 100, 64*4))

# Now get and plot the attention matrices from the `attentions` list
fig, axes = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        axes[i, j].imshow(attentions[i*2+j][0, 0])
        axes[i, j].set_title("Layer {} Head 0".format(i*2+j))
plt.tight_layout()
plt.show()
```

EventDispatcher
---------------

The event system is implemented by the [EventDispatcher][1] which is shared by
the transformer and attention modules as well as the rest of the system. The
event dispatcher instance is injected as an argument to all attentions and
transformer modules, but for further ease there is a global dictionary of event
dispatchers that is accessible through the `get(key="")` factory method as
follows:

```python
from fast_transformers.events import EventDispatcher

# The default dispatcher used by all modules unless passed as an argument
ed = EventDispatcher.get()
```

Unless an event dispatcher is provided via an argument, all modules simply use
the default event dispatcher.

### Methods

```python
EventDispatcher.listen(event_filter, event_handler)
```

The method `listen()` simply adds an event handler to be called when an event
is dispatched via this dispatcher. The event handler will only be called if the
event filter callable returns true for an event. For, details on the possible
values of `event_filter` see the [event filter][3] section.

The EventDispatcher automatically casts callables and Event subclasses to the
corresponding filter instances.

```python
EventDispatcher.dispatch(event)
```

Simply, call any event handler that is registered for this type of event.

```python
EventDispatcher.remove(event_handler)
EventDispatcher.clear()
```

Remove (unregister) a specific event handler using `remove()` or simply
unregister all of the event handlers using the `clear()` method of the event
dispatcher.

Event Filters
-------------

The event filters are callables that accept a single argument, an instance of
`Event`, and return whether to accept or dismiss this event. For ease of filter
composition, we provide an `EventFilter` object that allows for boolean
composition of filters using python operators, as follows:

```python
from fast_transformers.events.filters import event_class, from_layer, \
    layer_name_contains

# Checking whether an event is from a specific class
filter1 = event_class(AttentionEvent)

# Checking whether an event comes from a specific layer
filter2 = from_layer(net.layers[10])

# Checking whether the human readable name of the module contains a string
filter3 = layer_name_contains(net, "layers.10")

# Check whether it comes from a specific layer *and* is an AttentionEvent
filter4 = from_layer(net.layers[10] & event_class(AttentionEvent)
# or equivalently
filter4 = from_layer(net.layers[10] & AttentionEvent

# Check whether the attention matrix has 4 heads
filter5 = (
    event_class(AttentionEvent) &  # unless we also use the event_class
                                   # filter the event might not have the
                                   # attention_matrix attribute
    (lambda ev: ev.attention_matrix.shape[2]==4)
)
```

See the [event filters API docs][4] for more information.

Events
------

The events are subclasses of [Event][5] that contain the `source` layer from
which they were emitted and a payload that depends on the specific event that
was emitted.

The following is a list of the currently implemented events with a high-level
overview of their payload as well as the layers which emit them.

### <small>QKVEvent</small>

The QKVEvent is emmited by the [attention layer][6] and it contains the
`queries`, `keys` and `values` in the corresponding attributes.

### <small>AttentionEvent</small>

The AttentionEvent is emitted by the [full attention][7] and it contains the
softmax normalized attention matrix in the attribute `attention_matrix`.


[1]: /api_docs/fast_transformers/events/event_dispatcher.html
[2]: /api_docs/fast_transformers/events/index.html
[3]: events.md#event-filters
[4]: /api_docs/fast_transformers/events/filters.html
[5]: /api_docs/fast_transformers/events/event.html#fast_transformers.events.event.Event
[6]: /api_docs/fast_transformers/attention/attention_layer.html
[7]: /api_docs/fast_transformers/attention/full_attention.html
