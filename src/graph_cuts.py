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


# ---------------------------------------------------------------------------------------------
# Multi-tensor cuts: cut at ANY node boundary, even when several tensors cross it.
# The chunks are then node ranges, and the tensors that cross a cut are handed on by name.
# ---------------------------------------------------------------------------------------------

BLOCK_END_OPS = ("Mul", "Concat", "Add", "Relu", "MaxPool", "AveragePool", "Flatten", "Gemm", "Resize")


def tensor_usage(model):
    """(produced_at, last_use): node index that makes each tensor (-1 = graph input) and the last node that reads it."""
    nodes = list(model.graph.node)
    ignore = _ignored_tensors(model)
    produced_at = {i.name: -1 for i in model.graph.input if i.name not in ignore}
    for idx, nd in enumerate(nodes):
        for o in nd.output:
            if o and o not in ignore:
                produced_at[o] = idx
    last_use = {}
    for idx, nd in enumerate(nodes):
        for t in nd.input:
            if t and t not in ignore:
                last_use[t] = idx
    for o in model.graph.output:
        last_use[o.name] = len(nodes)
    return produced_at, last_use


def chunk_interfaces(model, cut_nodes):
    """
    For cuts placed after the given node indices, one (first_node, last_node, inputs, outputs) per chunk.
    inputs  = tensors the chunk reads that were made before it (or the graph input)
    outputs = tensors the chunk makes that later chunks, or the graph output, still need
    """
    nodes = list(model.graph.node)
    ignore = _ignored_tensors(model)
    produced_at, last_use = tensor_usage(model)
    bounds = [-1] + sorted(cut_nodes) + [len(nodes) - 1]
    chunks = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k] + 1, bounds[k + 1]
        inputs, seen = [], set()
        for idx in range(a, b + 1):
            for t in nodes[idx].input:
                if t and t not in ignore and t not in seen and produced_at.get(t, -1) < a:
                    seen.add(t)
                    inputs.append(t)
        outputs = sorted((t for t, pa in produced_at.items() if a <= pa <= b and last_use.get(t, -1) > b),
                         key=lambda t: (produced_at[t], t))
        chunks.append((a, b, inputs, outputs))
    return chunks


def node_work(model):
    """Rough work per node: 2 x multiply-adds for Conv/Gemm/MatMul, one unit per output element otherwise."""
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
    dims = {i.name: list(i.dims) for i in model.graph.initializer}
    work = []
    for nd in model.graph.node:
        out = shapes.get(nd.output[0]) if nd.output else None
        n_out = int(np.prod(out)) if out and all(out) else 0
        w = dims.get(nd.input[1]) if len(nd.input) > 1 else None
        if nd.op_type == "Conv" and w and n_out:
            work.append(2 * n_out * int(np.prod(w[1:])))             # [Cout, Cin/groups, kh, kw]
        elif nd.op_type in ("Gemm", "MatMul") and w and n_out:
            k = w[0] if (len(w) == 2 and out and w[1] == out[-1]) else (w[1] if len(w) == 2 else 1)
            work.append(2 * n_out * k)
        else:
            work.append(n_out)
    return work


def pick_balanced_cuts(model, k, after_ops=BLOCK_END_OPS):
    """k node indices (after block-ending nodes) that split the estimated work into k+1 roughly equal chunks."""
    nodes = list(model.graph.node)
    work = node_work(model)
    cum, run = [], 0
    for w in work:
        run += w
        cum.append(run)
    total = max(run, 1)
    allowed = [i for i in range(len(nodes) - 1) if nodes[i].op_type in after_ops]
    chosen = []
    for j in range(1, k + 1):
        target = total * j / (k + 1)
        best = min(allowed, key=lambda i: abs(cum[i] - target))
        if best not in chosen:
            chosen.append(best)
    return sorted(chosen), cum, total