"""
Task-set generation, following the reference paper's method (Section VIII-A):
UUniFast utilizations (Bini & Buttazzo 2005), D_i = T_i, Deadline-Monotonic
priorities, and WCET C_i = profiled time of the whole (unsplit) model.

Since U_i = C_i / T_i, each period follows from its utilization: T_i = C_i / U_i.

Practical deviation: a set is re-drawn if any task gets U_i < MIN_UTIL. A
task with a tiny utilization has a huge period and would release only a
handful of jobs in a short run, so it adds almost nothing to the observation.

Optional (snap_periods): round every period to a value from a fixed set whose
members all divide 120 base units. The whole task set then repeats exactly every
`hyperperiod_ms` (at most 120 base units), so "run for N hyperperiods", which the
paper does, is well defined. The price: the utilization actually achieved differs
a little from the target, and it is reported.

Usage:
    python experiments/generate_taskset.py
"""

import math
import random

MIN_UTIL = 0.03
PERIOD_MULTIPLES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]   # all divide 120
NICE_UNITS = sorted(m * 10 ** e for e in range(-1, 4) for m in (1, 2, 5))


def uunifast(n, total_u, rng):
    """n utilizations, uniformly distributed over {U_i >= 0, sum = total_u}."""
    utils, remaining = [], total_u
    for i in range(1, n):
        next_remaining = remaining * rng.random() ** (1.0 / (n - i))
        utils.append(remaining - next_remaining)
        remaining = next_remaining
    utils.append(remaining)
    return utils


def _name_by_priority(tasks):
    tasks.sort(key=lambda t: t["deadline_ms"])           # Deadline-Monotonic
    for prio, t in enumerate(tasks, start=1):
        t["priority"] = prio
        t["name"] = f"t{prio}"
    return tasks


def generate_task_set(n, total_u, wcet_ms, rng):
    """Returns n task dicts sorted by priority (priority 1 = highest)."""
    while True:
        utils = uunifast(n, total_u, rng)
        if min(utils) >= MIN_UTIL:
            break
    tasks = [{"util": u, "period_ms": wcet_ms / u, "deadline_ms": wcet_ms / u,
              "wcet_ms": wcet_ms} for u in utils]
    return _name_by_priority(tasks)


def base_unit_ms(wcet_ms):
    """Largest 'nice' number (1, 2, 5, 10, ...) not above wcet/3: the unit that periods are multiples of."""
    return max([u for u in NICE_UNITS if u <= wcet_ms / 3.0] or [NICE_UNITS[0]])


def snap_periods(tasks, wcet_ms):
    """
    Round each period to the nearest allowed value (nearest on a log scale). Allowed values
    are base_unit x PERIOD_MULTIPLES, and at least 5% above the WCET so no task needs more
    than 95% of a core by itself. Returns (tasks, hyperperiod_ms); each task gets util_actual.
    """
    unit = base_unit_ms(wcet_ms)
    allowed = [unit * m for m in PERIOD_MULTIPLES if unit * m >= 1.05 * wcet_ms]
    for t in tasks:
        target = t["period_ms"]
        t["period_ms"] = t["deadline_ms"] = min(allowed, key=lambda a: abs(math.log(a / target)))
        t["util_actual"] = wcet_ms / t["period_ms"]
    return _name_by_priority(tasks), unit * 120


if __name__ == "__main__":
    rng = random.Random(1)
    for total_u in (0.6, 0.9):
        tasks = generate_task_set(4, total_u, 2.8, rng)
        print(f"U = {total_u}: sum of utilizations = {sum(t['util'] for t in tasks):.3f}")
        for t in tasks:
            print(f"  {t['name']}  prio={t['priority']}  U={t['util']:.3f}  "
                  f"T=D={t['period_ms']:.2f}ms  C={t['wcet_ms']}ms")
    print("\nSame kind of set with snapped periods (C = 3.4 ms):")
    tasks, hyper = snap_periods(generate_task_set(4, 0.8, 3.4, random.Random(2)), 3.4)
    print(f"  hyperperiod <= {hyper:.0f} ms, achieved utilization {sum(t['util_actual'] for t in tasks):.3f}")
    for t in tasks:
        print(f"  {t['name']}  T=D={t['period_ms']:.1f}ms  U={t['util_actual']:.3f}")