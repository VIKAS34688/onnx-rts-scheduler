"""
All the result checks in one go (no GPU, reads files only). Run from the project root and paste the output:
    python experiments\\check_results.py

  1. heuristic/optimal sweeps: new run against the old run (did the chunk-file fix change anything?)
  2. detector Option B: first run against the rerun
  3. U=0.7 file: duplicate sets, and the two halves compared
  4. U=0.9 files: how far each run got
  5. the mixed-model quick test rows
  6. logs: Baseline SP demo, CPU-side times, and the stack file of the U=0.9 hang (if any)
"""

import csv
import os

RES = os.path.join("experiments", "results")
LOGS = os.path.join("docs", "phase2", "overnight_logs")


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(rows):
    acc = [r for r in rows if r["analysis_ok"] == "True"]
    ran = [r for r in acc if r.get("sched_misses", "") != ""]
    return {
        "rows": len(rows),
        "accepted": len(acc),
        "clean": sum(int(r["sched_misses"]) == 0 for r in ran),
        "base": sum(int(r["baseline_misses"]) for r in ran),
        "prop": sum(int(r["sched_misses"]) for r in ran),
        "preempt": sum(int(r["preemptions"] or 0) for r in ran),
    }


def fmt(s):
    return (f"rows {s['rows']:>3}  accepted {s['accepted']:>3}  zero-miss {s['clean']:>3}  "
            f"baseline misses {s['base']:>5}  proposed misses {s['prop']:>5}  preemptions {s['preempt']:>4}")


def compare(new_name, old_name):
    new, old = load(new_name), load(old_name)
    if new is None:
        print(f"  {new_name}: not found")
        return
    print(f"  {new_name}\n    new: {fmt(stats(new))}")
    if old is None:
        print("    old: not found")
        return
    print(f"    old: {fmt(stats(old))}")
    old_by_set = {(r["U"], r["set"]): r for r in old}
    diff_ok = diff_split = 0
    for r in new:
        o = old_by_set.get((r["U"], r["set"]))
        if o:
            diff_ok += r["analysis_ok"] != o["analysis_ok"]
            diff_split += r["split_points"] != o["split_points"]
    print(f"    sets where the analysis result differs: {diff_ok}, where the split differs: {diff_split}")


print("=== 1. heuristic / optimal sweeps, new against old ===")
for tag in ("h5", "o5"):
    for u in ("06", "07", "08", "09"):
        compare(f"alexnet_n3_{tag}_u{u}.csv", f"alexnet_n3_{tag}_u{u}_old.csv")

print("\n=== 2. YOLOv8n Option B: rerun against first run ===")
compare("yolo_n3_B_v2.csv", "yolo_n3_B.csv")

print("\n=== 3. U=0.7 file ===")
r = load("alexnet_n3_u07_rerun.csv")
if r:
    counts = {}
    for x in r:
        counts[x["set"]] = counts.get(x["set"], 0) + 1
    print(f"  rows {len(r)}, times each set appears: {sorted(set(counts.values()))}")
    half = len(r) // 2
    if len(r) % 2 == 0 and half:
        print(f"  first half : {fmt(stats(r[:half]))}")
        print(f"  second half: {fmt(stats(r[half:]))}")
else:
    print("  not found")

print("\n=== 4. U=0.9 files (the 7-candidate sweep that hangs) ===")
for name in ("alexnet_n3_u09.csv", "alexnet_n3_u09_part1.csv", "alexnet_n3_u09_part2.csv"):
    r = load(name)
    if r is None:
        print(f"  {name}: not found")
    else:
        last = r[-1]["set"] if r else "-"
        print(f"  {name}: {len(r)} rows, last finished set = {last}")

print("\n=== 5. mixed-model quick test ===")
r = load("mixed_quick_test.csv")
if r:
    for x in r:
        print(f"  U={x['U']} set={x['set']} [{x.get('models_in_set', '')}] analysis_ok={x['analysis_ok']} "
              f"baseline={x['baseline_misses']}/{x['baseline_jobs']} proposed={x['sched_misses']}/{x['sched_jobs']} "
              f"preemptions={x['preemptions']} split={x['split_points']}")
else:
    print("  not found")

print("\n=== 6. logs ===")
for name, limit in (("baseline_sp_demo.log", 30), ("cpu_side_times.log", 40),
                    ("alexnet_n3_u09.log.stack.txt", 60)):
    path = os.path.join(LOGS, name)
    print(f"--- {name}")
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        print("\n".join(lines[-limit:]))
    else:
        print("  not found")
