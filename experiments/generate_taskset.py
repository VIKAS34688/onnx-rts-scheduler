"""
Task-set generation, following the reference paper's method (Section VIII-A):
UUniFast utilizations (Bini & Buttazzo 2005), D_i = T_i, Deadline-Monotonic
priorities, and WCET C_i = profiled time of the whole (unsplit) model.

Since U_i = C_i / T_i, each period follows from its utilization: T_i = C_i / U_i.

Practical deviation: a set is re-drawn if any task gets U_i < MIN_UTIL. A
task with a tiny utilization has a huge period and would release only a
handful of jobs in a short run, so it adds almost nothing to the observation.

Usage:
    python experiments/generate_taskset.py
"""

import random

MIN_UTIL = 0.03


def uunifast(n, total_u, rng):
    """n utilizations, uniformly distributed over {U_i >= 0, sum = total_u}."""
    utils, remaining = [], total_u
    for i in range(1, n):
        next_remaining = remaining * rng.random() ** (1.0 / (n - i))
        utils.append(remaining - next_remaining)
        remaining = next_remaining
    utils.append(remaining)
    return utils


def generate_task_set(n, total_u, wcet_ms, rng):
    """Returns n task dicts sorted by priority (priority 1 = highest)."""
    while True:
        utils = uunifast(n, total_u, rng)
        if min(utils) >= MIN_UTIL:
            break
    tasks = [{"util": u, "period_ms": wcet_ms / u, "deadline_ms": wcet_ms / u,
              "wcet_ms": wcet_ms} for u in utils]
    tasks.sort(key=lambda t: t["deadline_ms"])           # Deadline-Monotonic
    for prio, t in enumerate(tasks, start=1):
        t["priority"] = prio
        t["name"] = f"t{prio}"
    return tasks


if __name__ == "__main__":
    rng = random.Random(1)
    for total_u in (0.6, 0.9):
        tasks = generate_task_set(4, total_u, 2.8, rng)
        print(f"U = {total_u}: sum of utilizations = {sum(t['util'] for t in tasks):.3f}")
        for t in tasks:
            print(f"  {t['name']}  prio={t['priority']}  U={t['util']:.3f}  "
                  f"T=D={t['period_ms']:.2f}ms  C={t['wcet_ms']}ms")
