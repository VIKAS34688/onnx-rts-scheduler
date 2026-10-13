"""
Re-runs ONE task set from the experiment several times, with Python's garbage
collector on and off, and prints when the late jobs happen. Used to find out
why a task set that the analysis accepts still shows a few deadline misses.

  - late jobs bunched together at one moment      -> a stall (OS or Python pause)
  - late jobs only in the first second            -> start-up effect
  - gc_off=True removes the misses                 -> Python garbage collection was the cause

Usage (same seed as the experiment, so it is the same task set):
    python experiments/diagnose_misses.py --u 0.9 --set 9
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import random

import schedulability as sch
import splitter
from _common import make_sample
from generate_taskset import generate_task_set
from run_baseline import run_baseline
from run_experiment import optimize_task_set, MODEL, CANDIDATE_IDX
from run_scheduled import run_scheduled

ap = argparse.ArgumentParser()
ap.add_argument("--u", type=float, required=True)
ap.add_argument("--set", type=int, required=True)
ap.add_argument("--n-tasks", type=int, default=4)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--duration", type=float, default=8.0)
args = ap.parse_args()

sch.load_profile_cache()
sample = make_sample(MODEL)
candidates = [splitter.list_candidate_split_points(MODEL)[i] for i in CANDIDATE_IDX]
c_whole = sch._profile_config(MODEL, (), sample, sch.PROFILE_CACHE, "models/chunks_opt")[0]

tasks = generate_task_set(args.n_tasks, args.u, c_whole, random.Random(f"{args.seed}-{args.u}-{args.set}"))
rts, ok = optimize_task_set(tasks, MODEL, candidates, sample, c_whole)
print(f"task set U={args.u} set={args.set}: analysis_ok={ok}")
for t, r in zip(tasks, rts or []):
    print(f"  {t['name']}  T=D={t['period_ms']:.2f}ms  chunks={len(r.chunk_wcets_ms)}  "
          f"analytic R={sch.response_time(r, rts):.2f}ms")


def show(label, res):
    print(f"{label}: misses={res['misses']}/{res['jobs']}  worst={res['worst_response_ms']:.1f}ms")
    for name, offset, resp, dl in res["late"][:8]:
        print(f"      late: {name} released at +{offset:.3f}s  response {resp}ms  (deadline {dl}ms)")


for rep in range(args.repeats):
    for gc_off in (False, True):
        show(f"proposed  rep={rep} gc_off={gc_off}",
             run_scheduled(tasks, rts, MODEL, candidates, args.duration, gc_off=gc_off))
for gc_off in (False, True):
    show(f"baseline  gc_off={gc_off}", run_baseline(tasks, MODEL, args.duration, gc_off=gc_off))