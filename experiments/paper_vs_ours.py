"""
Paper against ours: the comparison tables for the report (the paper's results R1 to R8).

It reads what the experiments already saved:
  docs/phase2/split_overhead_<model>.json   (split_overhead_table.py)   -> Tables II and IV
  the sweep CSVs you pass with --sweeps     (run_experiment.py)         -> Fig. 4 and Fig. 5
and prints one table per result. Anything not run yet shows as "not run yet", so you can run this
after every step and watch the table fill up. The text is also saved to docs/phase2/paper_vs_ours.md.

The paper's numbers are typed in below (from the paper's Tables II, IV, V and its text).
Results that no script saves (Fig. 1, memory, emit("Ours use the paper's number of cuts for each model (10, 5, 11 and 6).\n")start-up, dispatch cost) are in the MANUAL list:
edit the "ours" column there after each run.

Usage (project root):
    python experiments\\paper_vs_ours.py
    python experiments\\paper_vs_ours.py --sweeps experiments\\results\\alexnet_n3.csv experiments\\results\\alexnet_n3_optimal.csv
"""

import argparse
import csv
import json
import os
import statistics

# model: (split points found, split points used, overhead % fully split)   [paper Tables II and IV]
PAPER_SPLIT = {
    "alexnet":      (10, 10, 7.5),
    "inception_v4": (23, 6, 5.3),
    "resnet18":     (11, 11, 48.0),
    "vgg19":        (21, 5, 73.0),
}

# (id, what, paper, ours, note)  -- edit "ours" when you have new numbers
MANUAL = [
    ("R1", "Concurrent against sequential makespan (Fig. 1)",
     "similar makespans, sometimes worse; kernel concurrency as low as 1%",
     "set 1 (AlexNet, VGG19, ResNet18, InceptionV4): 49.13 ms concurrent vs 51.66 ms sequential, ratio 0.95; set 2 (GoogLeNet, MobileNetV2, VGG16, DenseNet121): 29.90 vs 33.96 ms, ratio 0.88; 500 runs each", ""),
    ("R6", "GPU memory, proposed against baseline (Table V)",
     "-1% to +2%",
     "one fully split AlexNet (8 chunk sessions) 530 MB against 544 MB for one whole-model session: -2.6%",
     "per task; our tasks with the same split share one chunk set, which saves more"),
    ("R7", "Start-up time",
     "at most 11.1 s (Jetson TX2)",
     "0.59 s for the 8-chunk set; 0.72 s for one whole-model session; 2.28 s for 4 whole-model sessions",
     "different hardware, so only the trend compares"),
    ("R8", "Dispatch cost per chunk",
     "8 to 13 microseconds",
     "gap between chunks: p50 0.08 to 0.11 ms, p99 0.18 to 0.35 ms (Phase 1)",
     "about 10x the paper's: Python plus ONNX Runtime calls on Windows"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="docs/phase2", help="folder with split_overhead_*.json")
ap.add_argument("--sweeps", nargs="*", default=[], help="sweep CSVs from run_experiment.py")
ap.add_argument("--out", default="docs/phase2/paper_vs_ours.md")
args = ap.parse_args()

lines = []


def emit(text=""):
    print(text)
    lines.append(text)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for r in rows:
        emit("| " + " | ".join(str(c) for c in r) + " |")
    emit()


# ---- R3 and R4: Tables II and IV ----------------------------------------------------------
emit("## R3 and R4: split points (Table II) and splitting overhead (Table IV)")
rows = []
for stem, (found, used, over) in PAPER_SPLIT.items():
    p = os.path.join(args.dir, f"split_overhead_{stem}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        ours = (f"{d['legal_cuts_found']} / {d['cuts_used']}", f"{d['overhead_pct_p50']:+.1f}%",
                f"{d['overhead_pct_p99']:+.1f}%")
    else:
        ours = ("not run yet",) * 3
    rows.append((stem, f"{found} / {used}", ours[0], f"+{over}%", ours[1], ours[2]))
table(["model", "paper found / used", "ours found / used", "paper overhead", "ours p50", "ours p99"], rows)
emit("Report the p50 overhead: the p99 figure adds up p99s of different sessions, which is noisy.")
emit("Ours use the paper's number of cuts for each model (10, 5, 11 and 6).\n")

# ---- R2 and R5: sweeps -----------------------------------------------------------------------
if args.sweeps:
    emit("## R2: schedulability against utilisation (Fig. 4)")
    emit("Paper: up to 96% against 10% for the baseline (Jetson Orin, 12 tasks, U=0.9).")
    emit("Ours: share of task sets with zero deadline misses. 'proposed' = accepted by the analysis AND zero misses when run.\n")
    runtime_rows = []
    for path in args.sweeps:
        rows_csv = list(csv.DictReader(open(path, newline="")))
        by_u = {}
        for r in rows_csv:
            by_u.setdefault(float(r["U"]), []).append(r)
        first = rows_csv[0]
        label = (f"{first.get('model', '?')}, {first.get('n_tasks', '?')} tasks, {first.get('method', '?')}"
                 + (", multi-tensor cuts" if first.get("multi") == "True" else ""))
        emit(f"**{os.path.basename(path)}** ({label})\n")
        out = []
        for U in sorted(by_u):
            rs = by_u[U]
            n = len(rs)
            acc = sum(r["analysis_ok"] == "True" for r in rs)
            val = sum(r["analysis_ok"] == "True" and r.get("sched_misses", "") != "" and int(r["sched_misses"]) == 0
                      for r in rs)
            base = sum(int(r["baseline_misses"]) == 0 for r in rs)
            out.append((U, n, f"{100 * base / n:.0f}%", f"{100 * acc / n:.0f}%", f"{100 * val / n:.0f}%"))
        table(["U", "sets", "baseline", "analysis accepts", "proposed (validated)"], out)

        def med(col):
            vals = [float(r[col]) for r in rows_csv if r.get(col, "") != ""]
            return f"{statistics.median(vals):.2f}" if vals else "n/a"
        runtime_rows.append((os.path.basename(path), first.get("method", "?"), med("opt_seconds"),
                             med("profile_seconds"), med("configs_evaluated")))
    emit("## R5: optimiser runtime (Fig. 5)")
    emit("Paper: the heuristic is up to 38% faster than the optimal search.\n")
    table(["file", "method", "median optimiser s", "median profiling s", "median configs"], runtime_rows)
else:
    emit("## R2 and R5: Fig. 4 and Fig. 5")
    emit("not run yet: pass the sweep CSVs with --sweeps.\n")

# ---- the rest ----------------------------------------------------------------------------------
emit("## R1, R6, R7, R8")
table(["id", "result", "paper", "ours", "note"], MANUAL)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"saved {args.out}")
