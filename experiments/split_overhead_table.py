"""
Paper results R3 and R4 (Tables II, III and IV) for any list of models:

  Table II : legal split points found, and how many we use
  Table III: size of the tensor that crosses each used cut
  Table IV : time of the whole model against the sum of its chunks when every used
             cut is active ("fully split"), i.e. the price of splitting

Usage (project root, GPU, close other apps first):
    python experiments/split_overhead_table.py models/alexnet.onnx models/vgg19.onnx
    python experiments/split_overhead_table.py models/resnet18.onnx --max-candidates 6 --runs 200
Results are also saved to docs/phase2/split_overhead_<model>.json
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import json
import os

import onnx

import profiler
import splitter
from _common import make_sample
from run_experiment import pick_candidates

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--max-candidates", type=int, default=7)
ap.add_argument("--runs", type=int, default=300)
args = ap.parse_args()

os.makedirs("docs/phase2", exist_ok=True)
summary = []
for path in args.models:
    stem = os.path.splitext(os.path.basename(path))[0]
    sample = make_sample(path)
    found = splitter.list_candidate_split_points(path)
    used = pick_candidates(path, args.max_candidates)
    print(f"\n=== {stem}: {len(found)} legal cuts found, {len(used)} used ===")

    inferred = onnx.shape_inference.infer_shapes(onnx.load(path))
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in inferred.graph.value_info}
    kb = []
    for t in used:
        shp = shapes.get(t, [])
        n = 1
        for d in shp:
            n *= d
        kb.append(round(n * 4 / 1024, 1) if shp and all(shp) else None)

    whole = profiler.profile_chunk(path, sample, n_runs=args.runs)
    paths = splitter.build_chunks(path, used, os.path.join("models", "chunks_tbl", stem))
    chunks = [profiler.profile_chunk(p, sample, n_runs=args.runs) for p in paths]
    s50, s99 = sum(c["p50_ms"] for c in chunks), sum(c["p99_ms"] for c in chunks)
    row = {
        "model": stem, "legal_cuts_found": len(found), "cuts_used": len(used), "chunks": len(paths),
        "whole_p50_ms": whole["p50_ms"], "whole_p99_ms": whole["p99_ms"],
        "split_sum_p50_ms": s50, "split_sum_p99_ms": s99,
        "overhead_pct_p50": 100 * (s50 - whole["p50_ms"]) / whole["p50_ms"],
        "overhead_pct_p99": 100 * (s99 - whole["p99_ms"]) / whole["p99_ms"],
        "chunk_p99_ms": [c["p99_ms"] for c in chunks], "chunk_p50_ms": [c["p50_ms"] for c in chunks],
        "cut_tensor_kb": kb,
    }
    json.dump(row, open(f"docs/phase2/split_overhead_{stem}.json", "w"), indent=1)
    summary.append(row)
    print(f"  whole model : p50 {whole['p50_ms']:.3f} ms   p99 {whole['p99_ms']:.3f} ms")
    print(f"  {len(paths)} chunks   : sum p50 {s50:.3f} ms   sum p99 {s99:.3f} ms")
    print(f"  overhead    : {row['overhead_pct_p50']:+.1f}% (p50)   {row['overhead_pct_p99']:+.1f}% (p99)")
    print(f"  chunk p99   : {[round(c['p99_ms'], 3) for c in chunks]} ms")
    print(f"  tensor size at the cuts (KB): {kb}")

print("\n=== Summary (compare with the paper's Tables II and IV) ===")
print(f"{'model':<14} {'found':>6} {'used':>5} {'whole p99':>10} {'split p99':>10} {'overhead p99':>13} {'overhead p50':>13}")
for r in summary:
    print(f"{r['model']:<14} {r['legal_cuts_found']:>6} {r['cuts_used']:>5} {r['whole_p99_ms']:>9.2f}ms "
          f"{r['split_sum_p99_ms']:>8.2f}ms {r['overhead_pct_p99']:>+12.1f}% {r['overhead_pct_p50']:>+12.1f}%")
