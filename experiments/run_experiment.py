"""
The experiment of the paper's Figure 4 on our hardware, for any ONNX model.

For each total utilization U and each random task set:
  1. optimize: split tasks from highest to lowest priority (Algorithm 2 greedy, or
     Algorithm 1 optimal), then check every task with the response-time analysis
  2. baseline: run the set unscheduled and unsplit, count deadline misses
  3. proposed: if the analysis accepts the set, run it through scheduler.py
     with the chosen chunks and count deadline misses (validation)

Chunk timings are measured once and saved (models/profile_cache.json), so later runs
reuse the SAME numbers. Use --cache-file to keep a separate cache, or delete the file.

Useful options:
  --model PATH         any ONNX model with one input and one output (default AlexNet)
  --max-candidates K   how many of the model's legal cut points the search may use
  --cand-idx I I ...   choose the candidate cuts yourself (indices into the legal-cut list)
  --method optimal     Algorithm 1 instead of the greedy Algorithm 2
  --snap               round periods to a fixed set so the task set has a hyperperiod, and
                       run for a whole number of hyperperiods (at least 10), as the paper does
  --fix-periods        build the periods from the raw WCET and add --overhead-ms only in the
                       analysis, so the real load does not change when the overhead changes

Usage (from the project root, close other apps first):
    python experiments/run_experiment.py --sets 2 --duration 4     (quick test)
    python experiments/run_experiment.py --model models/vgg19.onnx --n-tasks 3 --snap
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import csv
import math
import os
import random
import time

import schedulability as sch
import splitter
from _common import make_sample
from generate_taskset import generate_task_set, snap_periods
from run_baseline import run_baseline
from run_scheduled import run_scheduled
from schedulability import RTSTask, greedy_split, optimal_split, response_time

MODEL = os.environ.get("RTS_MODEL", "models/alexnet.onnx")
CANDIDATE_IDX = [2, 5, 7, 9, 12, 14, 16]     # AlexNet: 7 block boundaries (used by the Phase 1 results)

FIELDS = ["U", "set", "analysis_ok", "split_points", "baseline_jobs", "baseline_misses",
          "baseline_worst_ms", "sched_jobs", "sched_misses", "sched_worst_ms",
          "rule1_ok", "preemptions",
          "model", "method", "n_tasks", "U_actual", "hyper_ms", "duration_s",
          "configs_evaluated", "opt_seconds", "profile_seconds"]


def pick_candidates(model_path, k=7, cand_idx=None):
    """The cut points the search may use: explicit indices, the AlexNet default, or k evenly spread legal cuts."""
    points = splitter.list_candidate_split_points(model_path)
    if cand_idx:
        return [points[i] for i in cand_idx]
    if os.path.basename(model_path) == "alexnet.onnx" and k == 7:
        return [points[i] for i in CANDIDATE_IDX]
    if len(points) <= k:
        return points
    idx = sorted({min(len(points) - 1, max(0, int(round((j + 1) * (len(points) + 1) / (k + 1))) - 1))
                  for j in range(k)})
    return [points[i] for i in idx]


def optimize_task_set(tasks, model_path, candidates, sample, c_whole, method="heuristic"):
    """
    The paper's driver loop around Algorithm 1 or 2: highest priority first (never split),
    each next task is split just enough to fit the smallest blocking tolerance of the
    tasks above it. Then every task is checked with the response-time analysis.
    Returns (optimized RTSTask list or None, schedulable?).
    """
    split = optimal_split if method == "optimal" else greedy_split
    rts = []
    for t in tasks:                                   # already sorted by priority
        if not rts:
            task = RTSTask(t["name"], t["priority"], t["period_ms"], t["deadline_ms"], [c_whole])
            task.active_points = ()
        else:
            task = split(t["name"], t["priority"], t["period_ms"], t["deadline_ms"],
                         model_path, candidates, rts, sample)
            if task is None:
                return None, False
        rts.append(task)
    return rts, all(response_time(r, rts) <= r.deadline_ms for r in rts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-candidates", type=int, default=7)
    ap.add_argument("--cand-idx", type=int, nargs="+", default=None)
    ap.add_argument("--method", choices=["heuristic", "optimal"], default="heuristic")
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--fix-periods", action="store_true")
    ap.add_argument("--sets", type=int, default=10, help="task sets per utilization")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds per run")
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--utils", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overhead-ms", type=float, default=0.0,
                    help="per-chunk dispatcher cost added to every chunk WCET in the analysis")
    ap.add_argument("--gc-off", action="store_true", help="disable Python garbage collection during timed runs")
    ap.add_argument("--cache-file", default=sch.CACHE_FILE)
    ap.add_argument("--out", default="experiments/results/results.csv")
    args = ap.parse_args()
    sch.DISPATCH_OVERHEAD_MS = args.overhead_ms
    model = args.model

    n_cached = sch.load_profile_cache(args.cache_file)
    print(f"[INFO] loaded {n_cached} cached chunk measurements from {args.cache_file}")

    sample = make_sample(model)
    candidates = pick_candidates(model, args.max_candidates, args.cand_idx)
    print(f"[INFO] {os.path.basename(model)}: {len(splitter.list_candidate_split_points(model))} legal cuts, "
          f"{len(candidates)} used as candidates")
    c_whole = sch._profile_config(model, (), sample, sch.PROFILE_CACHE, "models/chunks_opt")[0]
    sch.save_profile_cache(args.cache_file)
    c_periods = c_whole - args.overhead_ms if args.fix_periods else c_whole
    print(f"[INFO] whole-model WCET ({sch.WCET_STAT}) = {c_whole:.3f} ms (includes the dispatch overhead); "
          f"periods are built from {c_periods:.3f} ms")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    fields = FIELDS
    if not new_file:
        with open(args.out, newline="") as f:
            fields = next(csv.reader(f))
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    if new_file:
        writer.writeheader()

    t_start = time.time()
    for U in args.utils:
        for s in range(args.sets):
            rng = random.Random(f"{args.seed}-{U}-{s}")
            tasks = generate_task_set(args.n_tasks, U, c_periods, rng)
            hyper_ms, duration = 0.0, args.duration
            if args.snap:
                tasks, hyper_ms = snap_periods(tasks, c_periods)
                duration = max(args.duration, 10 * hyper_ms / 1000.0)
                duration = math.ceil(duration * 1000.0 / hyper_ms) * hyper_ms / 1000.0
            u_actual = sum(t.get("util_actual", t["util"]) for t in tasks)

            sch.EVAL_KEYS.clear()
            t_opt, p_before = time.perf_counter(), sch.PROFILE_TIME_S
            rts, ok = optimize_task_set(tasks, model, candidates, sample, c_whole, args.method)
            opt_seconds = time.perf_counter() - t_opt
            profile_seconds = sch.PROFILE_TIME_S - p_before
            configs = len(sch.EVAL_KEYS)
            sch.save_profile_cache(args.cache_file)

            base = run_baseline(tasks, model, duration, gc_off=args.gc_off)
            sched = (run_scheduled(tasks, rts, model, candidates, duration, gc_off=args.gc_off)
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
                "model": os.path.basename(model), "method": args.method, "n_tasks": args.n_tasks,
                "U_actual": round(u_actual, 3), "hyper_ms": round(hyper_ms, 1), "duration_s": round(duration, 3),
                "configs_evaluated": configs, "opt_seconds": round(opt_seconds, 3),
                "profile_seconds": round(profile_seconds, 3),
            }
            writer.writerow(row)
            out.flush()
            print(f"U={U} set={s}: analysis_ok={ok}  baseline misses={base['misses']}/{base['jobs']}"
                  + (f"  proposed misses={sched['misses']}/{sched['jobs']} "
                     f"(rule1 {sched['rule1_ok']}, preemptions {sched['preemptions']})" if sched else "")
                  + f"   [{time.time() - t_start:.0f}s elapsed]")
    out.close()
    print(f"\n[DONE] results in {args.out}  ->  now run: python experiments/plot_results.py {args.out}")


if __name__ == "__main__":
    main()