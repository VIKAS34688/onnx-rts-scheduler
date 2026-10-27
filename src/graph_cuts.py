"""
Which places in an ONNX graph are legal cut points?

A cut after node i is legal when exactly ONE tensor crosses it: one tensor made
by nodes 0..i (or the graph input) that nodes after i still need. Then everything
before the cut can be saved as one chunk, everything after as the next, and that
single tensor is all that has to be handed over between them.

  straight chain (AlexNet)      -> every node boundary is legal
  residual block (ResNet)       -> NOT legal inside the block: the block input and
                                   the block's main path are both still needed
  backbone reused by a neck     -> NOT legal once a backbone feature map has been
  (YOLO, SSD)                      produced, because the neck needs it much later

ONNX stores nodes in a valid execution order, so we sweep through them once and
keep track of which tensors are still needed ("alive").

Constants and weights are ignored: they belong to one node and are stored with it.
"""

import numpy as np
import onnx


def _ignored_tensors(model):
    ignore = {i.name for i in model.graph.initializer}
    for nd in model.graph.node:
        if nd.op_type == "Constant":
            ignore.update(nd.output)
    return ignore


def alive_after_each_node(model):
    """List with one frozenset per node: the tensors that cross a cut placed right after that node."""
    nodes = list(model.graph.node)
    n = len(nodes)
    ignore = _ignored_tensors(model)

    last_use = {}
    for idx, nd in enumerate(nodes):
        for t in nd.input:
            if t and t not in ignore:
                last_use[t] = idx
    for o in model.graph.output:
        last_use[o.name] = n                      # needed at the very end

    alive = {i.name for i in model.graph.input if i.name not in ignore and last_use.get(i.name, -1) >= 0}
    result = []
    for idx, nd in enumerate(nodes):
        for o in nd.output:
            if o and o not in ignore:
                alive.add(o)
        alive = {t for t in alive if last_use.get(t, -1) > idx}
        result.append(frozenset(alive))
    return result


def legal_cut_points(model):
    """[(node_index, tensor_name), ...] for every legal cut, in graph order (no duplicates)."""
    nodes = list(model.graph.node)
    alive = alive_after_each_node(model)
    cuts, seen = [], set()
    for i in range(len(nodes) - 1):               # a cut after the last node is not a cut
        if len(alive[i]) == 1:
            (t,) = alive[i]
            if t not in seen:
                seen.add(t)
                cuts.append((i, t))
    return cuts


def node_weight_bytes(model):
    """Bytes of weights used by each node (a shared weight is counted at its first user)."""
    sizes = {}
    for init in model.graph.initializer:
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(init.data_type)).itemsize
        sizes[init.name] = int(np.prod(init.dims, dtype=np.int64)) * itemsize if init.dims else itemsize
    per_node, counted = [], set()
    for nd in model.graph.node:
        total = 0
        for t in nd.input:
            if t in sizes and t not in counted:
                counted.add(t)
                total += sizes[t]
        per_node.append(total)
    return per_node
