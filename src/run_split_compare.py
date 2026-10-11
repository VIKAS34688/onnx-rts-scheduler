"""
Run Algorithm 2 (greedy) and Algorithm 1 (optimal) on the SAME tau1 and
the same candidate split points, and compare them.

Usage:
    python src/run_split_compare.py
"""

import time
import torch  # import first -- registers PyTorch's bundled CUDA DLLs
import numpy as np

import splitter
import profiler
from schedulability import (RTSTask, greedy_split, optimal_split, response_time,
                            min_blocking_budget, SEARCH_RUNS, WCET_STAT)

MODEL = "models/alexnet.onnx"
DEADLINE_FACTOR = 1.6
sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

whole = profiler.profile_chunk(MODEL, sample, n_runs=SEARCH_RUNS)
c1 = whole[WCET_STAT]
D1 = T1 = round(DEADLINE_FACTOR * c1, 2)
tau1 = RTSTask("tau1", priority=1, period_ms=T1, deadline_ms=D1, chunk_wcets_ms=[c1])
B = min_blocking_budget([tau1])
print(f"[INFO] tau1: C1 = {c1:.3f}ms ({WCET_STAT}), T1 = D1 = {D1}ms, beta1 = {B:.3f}ms")

all_points = splitter.list_candidate_split_points(MODEL)
candidates = [all_points[i] for i in [2, 5, 7, 9, 12, 14, 16]]


def report(name, tau2, seconds):
    if tau2 is None:
        print(f"[{name}] not schedulable ({seconds:.0f}s)")
        return
    R = response_time(tau1, [tau1, tau2])
    print(f"\n[{name}] {len(tau2.active_points)} split points, "
          f"{len(tau2.chunk_wcets_ms)} chunks, search took {seconds:.0f}s")
    for p in tau2.active_points:
        print(f"   split after: {p}")
    print(f"   chunk WCETs (ms): {[round(w, 3) for w in tau2.chunk_wcets_ms]}")
    print(f"   total = {tau2.C:.3f}ms   largest = {tau2.C_max:.3f}ms   "
          f"R1 = {R:.3f}ms (D1 = {D1}ms)")


t0 = time.time()
g = greedy_split("tau2", 2, 50.0, 50.0, MODEL, candidates, [tau1], sample)
report("greedy", g, time.time() - t0)

print("\n[INFO] Running optimal search (15-30 minutes, do not use the laptop)...")
t0 = time.time()
o = optimal_split("tau2", 2, 50.0, 50.0, MODEL, candidates, [tau1], sample)
report("optimal", o, time.time() - t0)