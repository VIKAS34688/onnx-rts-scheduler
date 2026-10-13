"""
The Phase 1 experiment, mirroring the paper's Figure 4 on a smaller scale.

For each total utilization U and each random task set:
  1. optimize: split tasks from highest to lowest priority (Algorithm 2,
     greedy), then check every task with the response-time analysis
  2. baseline: run the set unscheduled and unsplit, count deadline misses
  3. proposed: if the analysis accepts the set, run it through scheduler.py
     with the chosen chunks and count deadline misses (validation)

Chunk timings are measured once and saved to models/profile_cache.json, so
later runs reuse the SAME numbers. Delete that file to re-measure.

Usage (from the project root, close other apps first):
    python experiments/run_experiment.py --sets 2 --duration 4     (quick test)
    python experiments/run_experiment.py                            (default run)
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import csv
import os
import random
import time

import schedulability as sch
import splitter
from _common import make_sample
from generate_taskset import generate_task_set
from run_baseline import run_baseline
from run_scheduled import run_scheduled
from schedulability import RTSTask, greedy_split, response_time

MODEL = "models/alexnet.onnx"
CANDIDATE_IDX = [2, 5, 7, 9, 12, 14, 16]     # the same 7 block boundaries as run_split_search.py


def optimize_task_set(tasks, model_path, candidates, sample, c_whole):
    """
    The paper's driver loop around Algorithm 2: highest priority first (never split),
    each next task is split just enough to fit the smallest blocking tolerance of the
    tasks above it. Then every task is checked with the response-time analysis.
    Returns (optimized RTSTask list or None, schedulable?).
    """
    rts = []
    for t in tasks:                                   # already sorted by priority
        if not rts:
            task = RTSTask(t["name"], t["priority"], t["period_ms"], t["deadline_ms"], [c_whole])
            task.active_points = ()
        else:
            task = greedy_split(t["name"], t["priority"], t["period_ms"], t["deadline_ms"],
                                model_path, candidates, rts, sample)
            if task is None:
                return None, False
        rts.append(task)
    return rts, all(response_time(r, rts) <= r.deadline_ms for r in rts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=10, help="task sets per utilization")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds per run")
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--utils", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gc-off", action="store_true", help="disable Python garbage collection during timed runs")
    ap.add_argument("--out", default="experiments/results/results.csv")
    args = ap.parse_args()

    n_cached = sch.load_profile_cache()
    print(f"[INFO] loaded {n_cached} cached chunk measurements")

    sample = make_sample(MODEL)
    candidates = [splitter.list_candidate_split_points(MODEL)[i] for i in CANDIDATE_IDX]
    c_whole = sch._profile_config(MODEL, (), sample, sch.PROFILE_CACHE, "models/chunks_opt")[0]
    sch.save_profile_cache()
    print(f"[INFO] whole-model WCET ({sch.WCET_STAT}) = {c_whole:.3f}ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["U", "set", "analysis_ok", "split_points", "baseline_jobs", "baseline_misses",
              "baseline_worst_ms", "sched_jobs", "sched_misses", "sched_worst_ms",
              "rule1_ok", "preemptions"]
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    if new_file:
        writer.writeheader()

    t_start = time.time()
    for U in args.utils:
        for s in range(args.sets):
            rng = random.Random(f"{args.seed}-{U}-{s}")
            tasks = generate_task_set(args.n_tasks, U, c_whole, rng)

            rts, ok = optimize_task_set(tasks, MODEL, candidates, sample, c_whole)
            sch.save_profile_cache()

            base = run_baseline(tasks, MODEL, args.duration, gc_off=args.gc_off)
            sched = (run_scheduled(tasks, rts, MODEL, candidates, args.duration, gc_off=args.gc_off)
                     if ok else None)

            row = {
                "U": U, "set": s, "analysis_ok": ok,
                "split_points": "/".join(str(len(r.active_points)) for r in rts) if rts else "",
                "baseline_jobs": base["jobs"], "baseline_misses": base["misses"],
                "baseline_worst_ms": round(base["worst_response_ms"], 3),
                "sched_jobs": sched["jobs"] if sched else "",
                "sched_misses": sched["misses"] if sched else "",
                "sched_worst_ms": round(sched["worst_response_ms"], 3) if sched else "",
                "rule1_ok": sched["rule1_ok"] if sched else "",
                "preemptions": sched["preemptions"] if sched else "",
            }
            writer.writerow(row)
            out.flush()
            print(f"U={U} set={s}: analysis_ok={ok}  baseline misses={base['misses']}/{base['jobs']}"
                  + (f"  proposed misses={sched['misses']}/{sched['jobs']} "
                     f"(rule1 {sched['rule1_ok']}, preemptions {sched['preemptions']})" if sched else "")
                  + f"   [{time.time() - t_start:.0f}s elapsed]")
    out.close()
    print(f"\n[DONE] results in {args.out}  ->  now run: python experiments/plot_results.py")


if __name__ == "__main__":
    main()