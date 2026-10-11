"""
Per-chunk timing diagnosis for the fully split AlexNet.
Shows p50 / p95 / p99 / max so we can see how much of the max is a
one-off spike, and profiles chunk 0 twice to test the idle-GPU-clock
explanation.

Usage:
    python src/diagnose_chunks.py
"""

import torch  # import first -- registers PyTorch's bundled CUDA DLLs
import numpy as np

import splitter
import profiler

MODEL = "models/alexnet.onnx"
RUNS = 300
sample = np.random.randn(1, 3, 224, 224).astype(np.float32)


def row(label, s, ends=""):
    print(f"{label:>9} {s['p50_ms']:>8.3f} {s['p95_ms']:>8.3f} "
          f"{s['p99_ms']:>8.3f} {s['max_ms']:>8.3f}   {ends}")


print(f"{'':>9} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
whole = profiler.profile_chunk(MODEL, sample, n_runs=RUNS)
row("whole", whole)

all_points = splitter.list_candidate_split_points(MODEL)
candidates = [all_points[i] for i in [2, 5, 7, 9, 12, 14, 16]]
paths = splitter.build_chunks(MODEL, candidates, "models/chunks_diag")

print()
sum_p50 = 0.0
for i, path in enumerate(paths):
    s = profiler.profile_chunk(path, sample, n_runs=RUNS)
    ends = candidates[i] if i < len(candidates) else "(model output)"
    row(f"chunk {i}", s, ends)
    sum_p50 += s["p50_ms"]

# chunk 0 again: if the first number was inflated by an idle GPU, this differs
s0 = profiler.profile_chunk(paths[0], sample, n_runs=RUNS)
row("chunk 0*", s0, "(profiled a second time)")

print(f"\nsum of chunk p50 = {sum_p50:.3f}ms   whole-model p50 = {whole['p50_ms']:.3f}ms")
print(f"splitting overhead = {sum_p50 - whole['p50_ms']:.3f}ms "
      f"({100 * (sum_p50 - whole['p50_ms']) / whole['p50_ms']:.1f}%)")