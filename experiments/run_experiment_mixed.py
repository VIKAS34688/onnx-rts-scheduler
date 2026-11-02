"""
Mixed-model experiment following the reference paper's protocol (Section VIII):
For each total utilization U and each random task set:
  1. Each task in the set uniformly picks a model from --models.
  2. Each task's period is built from the profiled whole-model WCET of ITS OWN model (p99).
  3. optimize: split tasks from highest to lowest priority (Algorithm 2 greedy or
     Algorithm 1 optimal), testing candidate cuts per model (from pick_candidates).
  4. baseline: run the mixed set concurrently and unscheduled with run_baseline_mixed.
  5. proposed: if the analysis accepts the set, run through scheduler.py (Rules 1-5)
     with run_scheduled_mixed.

Same CSV columns as run_experiment.py plus `models_in_set`.
"""

import _common  # noqa: F401  -- sets up sys.path for src/
import argparse
import csv
import math
import os
import random
import time

import schedulability as sch
import splitter
from _common import make_sample
from generate_taskset import uunifast
from generate_taskset_mixed import generate_mixed_task_set, snap_periods_mixed
from run_baseline_mixed import run_baseline_mixed
from run_scheduled_mixed import run_scheduled_mixed
from schedulability import RTSTask, greedy_split, optimal_split, response_time

DEFAULT_MODELS = ["models/alexnet.onnx", "models/resnet18.onnx"]
CANDIDATE_IDX = [2, 5, 7, 9, 12, 14, 16]

FIELDS = [
    "U", "set", "analysis_ok", "split_points", "baseline_jobs", "baseline_misses",
    "baseline_worst_ms", "sched_jobs", "sched_misses", "sched_worst_ms",
    "rule1_ok", "preemptions",
    "model", "models_in_set", "method", "multi", "n_tasks", "U_actual", "hyper_ms", "duration_s",
    "configs_evaluated", "opt_seconds", "profile_seconds"
]


def pick_candidates(model_path, k=7, cand_idx=None, multi=False):
    """The cut points the search may use for a given model."""
    if multi:
        return splitter.pick_multi_cuts(model_path, k)
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


def optimize_mixed_task_set(tasks, candidates_by_model, samples_by_model, c_whole_by_model,
                            method="heuristic", tmp_dir="models/chunks_opt"):
    """
    Split optimization across mixed models: highest priority first (never split),
    each subsequent task split just enough to fit the smallest blocking tolerance
    of higher-priority tasks.
    """
    split = optimal_split if method == "optimal" else greedy_split
    rts = []
    for t in tasks:
        m = t["model"]
        c_whole = c_whole_by_model[m]
        candidates = candidates_by_model[m]
        sample = samples_by_model[m]

        if not rts:
            task = RTSTask(t["name"], t["priority"], t["period_ms"], t["deadline_ms"], [c_whole])
            task.active_points = ()
        else:
            task = split(t["name"], t["priority"], t["period_ms"], t["deadline_ms"],
                         m, candidates, rts, sample, tmp_dir=tmp_dir)
            if task is None:
                return None, False
        rts.append(task)
    return rts, all(response_time(r, rts) <= r.deadline_ms for r in rts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="list of ONNX model paths to draw tasks from")
    ap.add_argument("--max-candidates", type=int, default=7)
    ap.add_argument("--cand-idx", type=int, nargs="+", default=None)
    ap.add_argument("--multi", action="store_true")
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
    ap.add_argument("--skip-speed-check", action="store_true")
    ap.add_argument("--out", default="experiments/results/results_mixed.csv")
    args = ap.parse_args()

    sch.DISPATCH_OVERHEAD_MS = args.overhead_ms
    models = [m.replace("\\", "/") for m in args.models]

    n_cached = sch.load_profile_cache(args.cache_file)
    print(f"[INFO] loaded {n_cached} cached chunk measurements from {args.cache_file}")

    samples_by_model = {}
    candidates_by_model = {}
    c_whole_by_model = {}
    c_periods_by_model = {}

    for m in models:
        sample = make_sample(m)
        samples_by_model[m] = sample
        cand = pick_candidates(m, args.max_candidates, args.cand_idx, args.multi)
        candidates_by_model[m] = cand
        print(f"[INFO] {os.path.basename(m)}: {len(splitter.list_candidate_split_points(m))} legal cuts, "
              f"{len(cand)} candidates used" + (" (multi-tensor)" if args.multi else ""))

        c_whole = sch._profile_config(m, (), sample, sch.PROFILE_CACHE, "models/chunks_opt")[0]
        c_whole_by_model[m] = c_whole

        if not args.skip_speed_check:
            import profiler
            now_p50 = profiler.profile_chunk(m, sample, n_runs=50)["p50_ms"]
            cached_raw = c_whole - args.overhead_ms
            if now_p50 > 1.25 * cached_raw:
                raise SystemExit(f"[ABORT] {m} takes {now_p50:.2f} ms now (median), but cached p99 is "
                                 f"{cached_raw:.2f} ms: device is running slow. Add --skip-speed-check to override.")
            print(f"[INFO] speed check passed for {os.path.basename(m)}: p50={now_p50:.2f}ms, cached p99={cached_raw:.2f}ms")

        c_periods = c_whole - args.overhead_ms if args.fix_periods else c_whole
        c_periods_by_model[m] = c_periods
        print(f"[INFO] {os.path.basename(m)} whole-model WCET ({sch.WCET_STAT}) = {c_whole:.3f} ms; "
              f"periods built from {c_periods:.3f} ms")

    sch.save_profile_cache(args.cache_file)

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
            tasks = generate_mixed_task_set(args.n_tasks, U, models, c_periods_by_model, rng)
            hyper_ms, duration = 0.0, args.duration
            if args.snap:
                tasks, hyper_ms = snap_periods_mixed(tasks, c_periods_by_model)
                duration = max(args.duration, 10 * hyper_ms / 1000.0)
                duration = math.ceil(duration * 1000.0 / hyper_ms) * hyper_ms / 1000.0
            u_actual = sum(t.get("util_actual", t["util"]) for t in tasks)

            sch.EVAL_KEYS.clear()
            t_opt, p_before = time.perf_counter(), sch.PROFILE_TIME_S
            rts, ok = optimize_mixed_task_set(tasks, candidates_by_model, samples_by_model,
                                              c_whole_by_model, method=args.method)
            opt_seconds = time.perf_counter() - t_opt
            profile_seconds = sch.PROFILE_TIME_S - p_before
            configs = len(sch.EVAL_KEYS)
            sch.save_profile_cache(args.cache_file)

            base = run_baseline_mixed(tasks, samples_by_model, duration, gc_off=args.gc_off)
            sched = (run_scheduled_mixed(tasks, rts, candidates_by_model, samples_by_model,
                                         duration, gc_off=args.gc_off)
                     if ok else None)

            models_in_set = "+".join(os.path.basename(t["model"]) for t in tasks)
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
                "model": "mixed",
                "models_in_set": models_in_set,
                "method": args.method, "multi": args.multi, "n_tasks": args.n_tasks,
                "U_actual": round(u_actual, 3), "hyper_ms": round(hyper_ms, 1), "duration_s": round(duration, 3),
                "configs_evaluated": configs, "opt_seconds": round(opt_seconds, 3),
                "profile_seconds": round(profile_seconds, 3),
            }
            writer.writerow(row)
            out.flush()
            print(f"U={U} set={s} [{models_in_set}]: analysis_ok={ok}  "
                  f"baseline misses={base['misses']}/{base['jobs']}"
                  + (f"  proposed misses={sched['misses']}/{sched['jobs']} "
                     f"(rule1 {sched['rule1_ok']}, preemptions {sched['preemptions']})" if sched else "")
                  + f"   [{time.time() - t_start:.0f}s elapsed]")

    out.close()
    print(f"\n[DONE] results in {args.out}")


if __name__ == "__main__":
    main()

