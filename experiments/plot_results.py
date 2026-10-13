"""
Reads experiments/results/results.csv and plots schedulability (%) against total
utilization, like the paper's Figure 4:

  Proposed (analysis)   task sets the response-time analysis accepts after splitting
  Proposed (validated)  accepted AND zero deadline misses when actually run
  Baseline (observed)   unsplit, unscheduled, zero deadline misses when actually run

Usage:
    python experiments/plot_results.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "experiments/results/results.csv"
PNG_PATH = "experiments/results/schedulability.png"


def truthy(v):
    return str(v).strip().lower() == "true"


rows = list(csv.DictReader(open(CSV_PATH)))
by_u = defaultdict(list)
for r in rows:
    by_u[float(r["U"])].append(r)

utils = sorted(by_u)
analysis, validated, baseline = [], [], []
print(f"{'U':>5} {'sets':>5} {'analysis':>9} {'validated':>10} {'baseline':>9}")
for u in utils:
    rs = by_u[u]
    n = len(rs)
    a = sum(truthy(r["analysis_ok"]) for r in rs)
    v = sum(truthy(r["analysis_ok"]) and r["sched_misses"] != "" and int(r["sched_misses"]) == 0 for r in rs)
    b = sum(int(r["baseline_misses"]) == 0 for r in rs)
    analysis.append(100.0 * a / n)
    validated.append(100.0 * v / n)
    baseline.append(100.0 * b / n)
    print(f"{u:>5} {n:>5} {a:>9} {v:>10} {b:>9}")

plt.figure(figsize=(6.5, 4.2))
plt.plot(utils, analysis, "o-", label="Proposed (analysis)")
plt.plot(utils, validated, "s--", label="Proposed (validated by run)")
plt.plot(utils, baseline, "^-", label="Baseline (observed)")
plt.xlabel("Total utilization U")
plt.ylabel("Schedulable task sets (%)")
plt.ylim(-3, 103)
plt.xticks(utils)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Schedulability vs. utilization (AlexNet, ONNX Runtime)")
plt.tight_layout()
os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
plt.savefig(PNG_PATH, dpi=150)
print(f"\n[DONE] saved {PNG_PATH}")