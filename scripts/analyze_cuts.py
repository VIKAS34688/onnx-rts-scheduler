"""
How splittable is an ONNX model? Lists every legal cut point and shows how much
of the model sits between them, so you can see BEFORE profiling whether
splitting can make the chunks small.

Usage (project root, venv active):
    python scripts/analyze_cuts.py models/alexnet.onnx
    python scripts/analyze_cuts.py models/yolov8n.onnx

"% of nodes / weights before" are proxies for how much work lies before a cut.
Real chunk times come from the profiler.
"""

import os
import sys
from collections import Counter

import onnx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from graph_cuts import legal_cut_points, node_weight_bytes

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/analyze_cuts.py MODEL.onnx")
path = sys.argv[1]

model = onnx.load(path)
inferred = onnx.shape_inference.infer_shapes(model)
shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in inferred.graph.value_info}
nodes = list(model.graph.node)
n = len(nodes)
w = node_weight_bytes(model)
total_w = sum(w) or 1

print(f"model : {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
print(f"input : {model.graph.input[0].name} {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
print(f"outputs: " + ", ".join(f"{o.name} {[d.dim_value for d in o.type.tensor_type.shape.dim]}" for o in model.graph.output))
ops = Counter(nd.op_type for nd in nodes)
print(f"nodes : {n}   " + ", ".join(f"{k} x{v}" for k, v in ops.most_common(8)))

cuts = legal_cut_points(model)
print(f"\nlegal cuts: {len(cuts)} of {n - 1} node boundaries\n")
if cuts:
    print(f"{'#':>3} {'after node':>10} {'op':<12} {'tensor':<34} {'shape':<18} {'KB':>8} {'% nodes':>8} {'% weights':>9}")
    cum_w = [0] * n
    run = 0
    for i in range(n):
        run += w[i]
        cum_w[i] = run
    for k, (i, t) in enumerate(cuts):
        shp = shapes.get(t, [])
        kb = (1.0 * eval("*".join(map(str, shp))) * 4 / 1024) if shp and all(shp) else float("nan")
        print(f"{k:>3} {i:>10} {nodes[i].op_type:<12} {t[-34:]:<34} {str(shp):<18} {kb:>8.1f} "
              f"{100 * (i + 1) / n:>7.0f}% {100 * cum_w[i] / total_w:>8.0f}%")

bounds = [-1] + [i for i, _ in cuts] + [n - 1]
segments = [(bounds[k] + 1, bounds[k + 1]) for k in range(len(bounds) - 1)]
seg_nodes = [b - a + 1 for a, b in segments]
seg_w = [sum(w[a:b + 1]) for a, b in segments]
big = max(range(len(segments)), key=lambda k: seg_nodes[k])
bigw = max(range(len(segments)), key=lambda k: seg_w[k])
print(f"\nsegments between legal cuts: {len(segments)}")
print(f"largest by nodes  : nodes {segments[big][0]}..{segments[big][1]}  = {seg_nodes[big]} nodes "
      f"({100 * seg_nodes[big] / n:.0f}% of the model)")
print(f"largest by weights: nodes {segments[bigw][0]}..{segments[bigw][1]}  = {seg_w[bigw] / 1e6:.1f} MB "
      f"({100 * seg_w[bigw] / total_w:.0f}% of the weights)")
print("\nRule of thumb: the blocking one chunk can cause is at least the time of the largest")
print("segment that has no legal cut inside it, so a large share here limits what splitting can do.")
