"""
First end-to-end run of the Person B pipeline on real AlexNet:
split -> profile -> check against blocking tolerance -> repeat (Algorithm 2).

Scenario (a stress test to exercise the algorithm, not the final experiment):
  tau1 = high-priority task running the WHOLE AlexNet (never split)
  tau2 = low-priority task running AlexNet, whose split we search for
tau1's blocking tolerance is beta = D1 - C1, so every chunk of tau2 must
be shorter than that, or tau1 can miss its deadline.

WCET estimator: p99 over 300 runs (schedulability.WCET_STAT), applied to
tau1 and to every chunk alike. The paper uses the observed max on dedicated
embedded boards; on a general-purpose Windows laptop the max is dominated by
one-off OS stalls (see diagnose_chunks.py), so we deviate openly and report
the max alongside it.

tau1's deadline is scaled to the measured speed of AlexNet on this GPU:
  - unsplit tau2 blocks tau1 for about C1, so R1 is about 2*C1 -> fails when D1 < 2*C1
  - a split tau2 passes once its largest chunk fits inside beta = D1 - C1
A factor between 1 and 2 therefore gives a meaningful demo.

Usage:
    python src/run_split_search.py
"""

import time
import torch  # import first -- registers PyTorch's bundled CUDA DLLs
import numpy as np

import splitter
import profiler
from schedulability import (RTSTask, greedy_split, response_time,
                            min_blocking_budget, SEARCH_RUNS, WCET_STAT)

MODEL = "models/alexnet.onnx"
DEADLINE_FACTOR = 1.6   # D1 = T1 = DEADLINE_FACTOR * C1 (must be between 1 and 2)
sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

# --- tau1: the high-priority task ---
print(f"[INFO] Profiling unsplit AlexNet (tau1's WCET = {WCET_STAT} over {SEARCH_RUNS} runs)...")
whole = profiler.profile_chunk(MODEL, sample, n_runs=SEARCH_RUNS)
c1 = whole[WCET_STAT]
print(f"[INFO] whole model: p50={whole['p50_ms']:.3f}ms  p95={whole['p95_ms']:.3f}ms  "
      f"p99={whole['p99_ms']:.3f}ms  max={whole['max_ms']:.3f}ms")
D1 = T1 = round(DEADLINE_FACTOR * c1, 2)
tau1 = RTSTask("tau1", priority=1, period_ms=T1, deadline_ms=D1, chunk_wcets_ms=[c1])
B = min_blocking_budget([tau1])
print(f"[INFO] tau1: C1 = {c1:.3f}ms, T1 = D1 = {D1}ms")
print(f"[INFO] Blocking budget for tau2's chunks: beta1 = {B:.3f}ms")

# --- what happens if tau2 is NOT split ---
tau2_unsplit = RTSTask("tau2", 2, 50.0, 50.0, [c1])
R_unsplit = response_time(tau1, [tau1, tau2_unsplit])
print(f"[INFO] Unsplit tau2 -> R1 = {R_unsplit:.3f}ms "
      f"(deadline {D1}ms) schedulable: {R_unsplit <= D1}")

# --- candidate split points: a SHORT list of block boundaries ---
all_points = splitter.list_candidate_split_points(MODEL)
candidate_idx = [2, 5, 7, 9, 12, 14, 16]
candidates = [all_points[i] for i in candidate_idx]
print(f"[INFO] Searching over {len(candidates)} candidate split points:")
for c in candidates:
    print(f"   {c}")

# --- Algorithm 2 (greedy) ---
print("\n[INFO] Running greedy split search (a few minutes)...")
t0 = time.time()
tau2 = greedy_split("tau2", 2, 50.0, 50.0, MODEL, candidates, [tau1], sample)
print(f"[INFO] Search finished in {time.time() - t0:.1f}s")

if tau2 is None:
    print("[RESULT] Not schedulable even fully split: the largest chunk is bigger "
          "than beta1. Raise DEADLINE_FACTOR (for example 1.8) and re-run.")
else:
    n_chunks = len(tau2.chunk_wcets_ms)
    print(f"\n[RESULT] Greedy chose {len(tau2.active_points)} split points "
          f"-> {n_chunks} chunks")
    for p in tau2.active_points:
        print(f"   split after: {p}")
    print(f"   chunk WCETs ({WCET_STAT}, ms):", [round(w, 3) for w in tau2.chunk_wcets_ms])
    print(f"   largest chunk = {tau2.C_max:.3f}ms (must be <= beta1 = {B:.3f}ms)")
    print(f"   total split WCET = {tau2.C:.3f}ms vs unsplit {c1:.3f}ms "
          f"-> splitting overhead = {tau2.C - c1:.3f}ms")

    R = response_time(tau1, [tau1, tau2])
    print(f"   tau1 response time with split tau2: R1 = {R:.3f}ms "
          f"(deadline {D1}ms) schedulable: {R <= D1}")

    # correctness: do the chosen chunks still reproduce the original output?
    paths = splitter.build_chunks(MODEL, list(tau2.active_points), "models/chunks_final")
    ok = splitter.verify_chunks_match_original(MODEL, paths, sample)
    print(f"[{'PASS' if ok else 'FAIL'}] Chosen chunks match original model output: {ok}")