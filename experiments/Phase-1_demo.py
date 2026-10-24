"""
Live demo for the viva: 4 tasks, each split into 4 chunks, run through the
priority scheduler on one GPU.

What it does, in order:
  1. cuts AlexNet into 4 chunks and measures each chunk's worst-case time
  2. builds 4 periodic tasks (task 1 = highest priority) and checks them with the
     response-time analysis (Eq. 1 and 2)
  3. runs the same 4 tasks WITHOUT scheduling or splitting (baseline)
  4. runs them through scheduler.py (Rules 1 to 5)
  5. prints per-task results and draws a timeline (Gantt chart) of which chunk
     ran when, so you can SEE a low-priority job give way at a chunk boundary

Usage (project root, charger in, other apps closed):
    python experiments/demo.py
    python experiments/demo.py --scale 0.7        (shorter periods = heavier load)
    python experiments/demo.py --no-baseline      (skip the baseline run)
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import math
import os
import sys
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort

import profiler
import schedulability as sch
import splitter
from _common import make_sample, sleep_until, summarize, warm_for
from device import DEVICE, PROVIDERS
from run_baseline import run_baseline
from run_experiment import MODEL, CANDIDATE_IDX
from run_scheduled import _warm_up, chunk_paths_for
from scheduler import Scheduler, Task, verify_no_overlap

CUT_IDX = (7, 14, 16)                 # 3 cuts -> 4 chunks, balanced in time
PERIOD_FACTORS = (3.2, 4.8, 8.0, 16.0)  # period of task i = factor x (job cost in the analysis)

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=4.0, help="seconds per run")
ap.add_argument("--scale", type=float, default=1.0, help="multiplies all periods (below 1 = heavier load)")
ap.add_argument("--overhead-ms", type=float, default=0.35, help="dispatcher cost added per chunk in the analysis")
ap.add_argument("--profile-runs", type=int, default=100)
ap.add_argument("--window-ms", type=float, default=40.0, help="length of the timeline that is drawn")
ap.add_argument("--no-baseline", action="store_true")
ap.add_argument("--out", default="experiments/results/demo_timeline.png")
args = ap.parse_args()

# ---------------------------------------------------------------- 1. chunks
all_points = splitter.list_candidate_split_points(MODEL)
candidates = [all_points[i] for i in CANDIDATE_IDX]
cuts = tuple(all_points[i] for i in CUT_IDX)
paths = chunk_paths_for(MODEL, candidates, cuts)
sample = make_sample(MODEL)

print("\n=== 1. AlexNet cut into 4 chunks (worst-case time = p99 of %d runs) ===" % args.profile_runs)
wcet = [profiler.profile_chunk(p, sample, n_runs=args.profile_runs)["p99_ms"] for p in paths]
cost = [w + args.overhead_ms for w in wcet]          # analysis cost of a chunk = compute + dispatcher
for i, (w, c) in enumerate(zip(wcet, cost)):
    print(f"  chunk {i + 1}: compute {w:.3f} ms  + dispatch {args.overhead_ms:.2f} ms = {c:.3f} ms")
job_cost = sum(cost)
print(f"  one job = {job_cost:.3f} ms in the analysis, largest chunk = {max(cost):.3f} ms")

# ---------------------------------------------------------------- 2. tasks + analysis
tasks = []
for i, f in enumerate(PERIOD_FACTORS):
    period = f * job_cost * args.scale
    tasks.append({"name": f"task {i + 1}", "priority": i + 1, "period_ms": period, "deadline_ms": period})
rts = [sch.RTSTask(t["name"], t["priority"], t["period_ms"], t["deadline_ms"], cost) for t in tasks]
bound = {r.name: sch.response_time(r, rts) for r in rts}

print("\n=== 2. The task set (priority 1 = highest, deadline = period) ===")
print(f"  {'task':<7} {'priority':>8} {'period':>10} {'4 chunks':>9} {'analysis bound R':>17} {'meets deadline?':>16}")
for t in tasks:
    ok = bound[t["name"]] <= t["deadline_ms"]
    print(f"  {t['name']:<7} {t['priority']:>8} {t['period_ms']:>8.1f}ms {'yes':>9} "
          f"{bound[t['name']]:>14.2f} ms {'yes' if ok else 'NO':>16}")
util = sum(job_cost / t["period_ms"] for t in tasks)
print(f"  utilisation in the analysis: {util:.2f}")

# ---------------------------------------------------------------- 3. baseline
if not args.no_baseline:
    print(f"\n=== 3. Baseline: same tasks, whole model, no priorities, no chunks ({args.duration:.0f} s) ===")
    base = run_baseline(tasks, MODEL, args.duration)
    print(f"  deadline misses: {base['misses']} of {base['jobs']} jobs, worst response {base['worst_response_ms']:.1f} ms")
    for name, m in base["misses_per_task"].items():
        print(f"    {name}: {m} misses")

# ---------------------------------------------------------------- 4. scheduled run
print(f"\n=== 4. Scheduled: same tasks, 4 chunks each, priority scheduler ({args.duration:.0f} s) ===")
sessions = [ort.InferenceSession(p, providers=PROVIDERS) for p in paths]
x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
in_name = sessions[0].get_inputs()[0].name
feed = lambda chunk_index, prev: {in_name: x}
warm_for(1.0, lambda: _warm_up(sessions, x, 1))

scheduler = Scheduler()
scheduler.start()
records, lock = [], threading.Lock()
total_jobs = sum(math.ceil(args.duration / (t["period_ms"] / 1000.0)) for t in tasks)
t0 = time.perf_counter() + 0.05


def make_callback(name, release):
    def done(finish):
        with lock:
            records.append((name, release, finish))
    return done


def release_loop(t):
    period = t["period_ms"] / 1000.0
    for k in range(math.ceil(args.duration / period)):
        release = t0 + k * period
        sleep_until(release)
        scheduler.submit_job(Task(t["name"], t["priority"], sessions, feed,
                                  on_complete=make_callback(t["name"], release)))


threads = [threading.Thread(target=release_loop, args=(t,)) for t in tasks]
for th in threads:
    th.start()
for th in threads:
    th.join()
deadline = time.perf_counter() + 20.0
while time.perf_counter() < deadline:
    with lock:
        if len(records) >= total_jobs:
            break
    time.sleep(0.05)
scheduler.stop()

res = summarize(records, tasks)
worst = {t["name"]: 0.0 for t in tasks}
njobs = {t["name"]: 0 for t in tasks}
for name, rel, fin in records:
    worst[name] = max(worst[name], (fin - rel) * 1000.0)
    njobs[name] += 1
print(f"  deadline misses: {res['misses']} of {res['jobs']} jobs, worst response {res['worst_response_ms']:.1f} ms")
print(f"  {'task':<7} {'jobs':>5} {'worst seen':>11} {'analysis bound':>15} {'deadline':>10} {'misses':>7}")
for t in tasks:
    n = t["name"]
    print(f"  {n:<7} {njobs[n]:>5} {worst[n]:>8.2f} ms {bound[n]:>12.2f} ms {t['deadline_ms']:>7.1f} ms {res['misses_per_task'][n]:>7}")
print(f"  Rule 1 (one chunk at a time):   {'PASS' if verify_no_overlap(scheduler.execution_log) else 'FAIL'}")
print(f"  Rule 5 (job gave way at a chunk boundary): {scheduler.preemptions} times")

# ---------------------------------------------------------------- 5. timeline
log = sorted(scheduler.execution_log, key=lambda e: e["start"])
win0 = t0 + 1.0                       # skip the first second
win1 = win0 + args.window_ms / 1000.0
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
rank = {t["name"]: i for i, t in enumerate(tasks)}

fig, ax = plt.subplots(figsize=(11, 3.6))
for idx, e in enumerate(log):
    if e["end"] < win0 or e["start"] > win1:
        continue
    row = 3 - rank[e["task_id"]]
    start_ms, dur_ms = (e["start"] - win0) * 1000.0, (e["end"] - e["start"]) * 1000.0
    k = e["chunk_index"]
    prev = log[idx - 1] if idx > 0 else None
    resumed = k > 0 and not (prev and prev["task_id"] == e["task_id"] and prev["chunk_index"] == k - 1)
    ax.broken_barh([(start_ms, dur_ms)], (row - 0.35, 0.7), facecolors=colors[rank[e["task_id"]]],
                   edgecolor="red" if resumed else "black", linewidth=1.6 if resumed else 0.6, alpha=0.5 + 0.12 * k)
    if dur_ms > 0.45:
        ax.text(start_ms + dur_ms / 2, row, str(k + 1), ha="center", va="center", fontsize=8, color="black")
for t in tasks:
    row, period = 3 - rank[t["name"]], t["period_ms"] / 1000.0
    for k in range(math.ceil(args.duration / period)):
        rel = t0 + k * period
        if win0 <= rel <= win1:
            ax.plot((rel - win0) * 1000.0, row - 0.45, marker="^", color="black", markersize=6)
ax.set_xlim(0, args.window_ms)
ax.set_ylim(-0.6, 3.6)
ax.set_yticks([3, 2, 1, 0])
ax.set_yticklabels([f"{t['name']} (prio {t['priority']})" for t in tasks])
ax.set_xlabel("time (ms)")
ax.set_title("One GPU, 4 tasks x 4 chunks. Numbers = chunk 1..4. Triangle = job released. "
             "Red outline = job resumed after giving way.", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f"\n[DONE] timeline saved to {args.out}")
if sys.platform == "win32":
    try:
        os.startfile(os.path.abspath(args.out))
    except OSError:
        pass
