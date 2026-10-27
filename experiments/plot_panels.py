"""
Draws the paper's Figure 4 layout: one panel per results CSV (schedulability against
utilization), so several task counts or models sit side by side.

  Proposed (analysis)   task sets the response-time analysis accepts after splitting
  Proposed (validated)  accepted AND zero deadline misses when actually run
  Baseline (observed)   unsplit, unscheduled, zero deadline misses when actually run

Usage:
    python experiments/plot_panels.py results_a.csv results_b.csv results_c.csv --out docs/phase2/figure4.png
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--out", default="experiments/results/panels.png")
ap.add_argument("--cols", type=int, default=3)
args = ap.parse_args()


def truthy(v):
    return str(v).strip().lower() == "true"


n = len(args.csvs)
cols = min(args.cols, n)
rows_n = math.ceil(n / cols)
fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.3 * rows_n), squeeze=False, sharey=True)
for ax, path in zip(axes.flat, args.csvs):
    rows = list(csv.DictReader(open(path)))
    by_u = defaultdict(list)
    for r in rows:
        by_u[float(r["U"])].append(r)
    us = sorted(by_u)
    a_, v_, b_ = [], [], []
    for u in us:
        rs = by_u[u]
        k = len(rs)
        a_.append(100.0 * sum(truthy(r["analysis_ok"]) for r in rs) / k)
        v_.append(100.0 * sum(truthy(r["analysis_ok"]) and r["sched_misses"] != "" and int(r["sched_misses"]) == 0 for r in rs) / k)
        b_.append(100.0 * sum(int(r["baseline_misses"]) == 0 for r in rs) / k)
    ax.plot(us, a_, "o-", label="Proposed (analysis)")
    ax.plot(us, v_, "s--", label="Proposed (validated)")
    ax.plot(us, b_, "^-", label="Baseline (observed)")
    first = rows[0]
    title = (f"{first.get('n_tasks', '?')} tasks, {str(first.get('model', os.path.basename(path))).replace('.onnx', '')}"
             f", {first.get('method', '')}".rstrip(", "))
    ax.set_title(f"{title}  ({len(rows) // len(us)} sets/point)", fontsize=9)
    ax.set_xticks(us)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.set_xlabel("total utilization U", fontsize=8)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
for r in range(rows_n):
    axes[r][0].set_ylabel("schedulable task sets (%)", fontsize=8)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f"[DONE] saved {args.out}")
