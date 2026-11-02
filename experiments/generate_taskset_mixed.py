"""
Task-set generation for mixed models, following the reference paper's protocol
(Section VIII-A):
- Each task in a set picks its model uniformly from the provided `--models` list.
- UUniFast utilizations (Bini & Buttazzo 2005) with min utilization >= MIN_UTIL.
- D_i = T_i (implicit deadlines) and Deadline-Monotonic priorities.
- Each task's period is built from the profiled whole-model WCET of ITS OWN model:
      T_i = D_i = C_i / U_i.
- Optional snap_periods_mixed: round each period to a value from allowed multiples
  sharing a common base unit so the entire mixed task set has a well-defined hyperperiod.
"""

import math
import random
from generate_taskset import MIN_UTIL, PERIOD_MULTIPLES, NICE_UNITS, uunifast, _name_by_priority


def base_unit_mixed_ms(wcets_ms):
    """
    Find a common base unit in NICE_UNITS suitable for all models in the set.
    Must not exceed min(wcet)/3 and must allow the largest model's period to fit
    within 120 base units.
    """
    min_c = min(wcets_ms)
    max_c = max(wcets_ms)
    candidates = [u for u in NICE_UNITS if u <= min_c / 3.0]
    unit = max(candidates) if candidates else NICE_UNITS[0]

    # Ensure 120 * unit is at least 1.05 * max_c so every task has at least one valid period
    while unit * 120 < 1.05 * max_c:
        idx = NICE_UNITS.index(unit)
        if idx + 1 < len(NICE_UNITS):
            unit = NICE_UNITS[idx + 1]
        else:
            break
    return unit


def generate_mixed_task_set(n, total_u, models, c_periods_by_model, rng):
    """
    Returns n task dicts with uniformly chosen models, sorted by Deadline-Monotonic priority.
    """
    while True:
        utils = uunifast(n, total_u, rng)
        if min(utils) >= MIN_UTIL:
            break

    tasks = []
    for u in utils:
        m = rng.choice(models)
        wcet = c_periods_by_model[m]
        tasks.append({
            "model": m,
            "util": u,
            "wcet_ms": wcet,
            "period_ms": wcet / u,
            "deadline_ms": wcet / u,
        })
    return _name_by_priority(tasks)


def snap_periods_mixed(tasks, c_periods_by_model):
    """
    Round each period in a mixed-model task set to the nearest allowed value
    (nearest on a log scale) sharing a common base unit.
    Returns (tasks, hyperperiod_ms).
    """
    wcets = [t["wcet_ms"] for t in tasks]
    unit = base_unit_mixed_ms(wcets)

    for t in tasks:
        wcet = t["wcet_ms"]
        allowed = [unit * m for m in PERIOD_MULTIPLES if unit * m >= 1.05 * wcet]
        if not allowed:
            # Fallback in extreme cases: allow closest multiple without 5% headroom check
            allowed = [unit * m for m in PERIOD_MULTIPLES]
        target = t["period_ms"]
        best = min(allowed, key=lambda a: abs(math.log(a / target)))
        t["period_ms"] = t["deadline_ms"] = best
        t["util_actual"] = wcet / t["period_ms"]

    multiples = [max(1, int(round(t["period_ms"] / unit))) for t in tasks]
    hyper_ms = math.lcm(*multiples) * unit
    return _name_by_priority(tasks), hyper_ms


if __name__ == "__main__":
    rng = random.Random(1)
    models = ["models/alexnet.onnx", "models/resnet18.onnx"]
    c_map = {"models/alexnet.onnx": 2.8, "models/resnet18.onnx": 6.5}
    tasks = generate_mixed_task_set(4, 0.7, models, c_map, rng)
    print("Raw mixed task set:")
    for t in tasks:
        print(f"  {t['name']} prio={t['priority']} model={t['model']} U={t['util']:.3f} T={t['period_ms']:.2f}ms")

    snapped, hyper = snap_periods_mixed(tasks, c_map)
    print(f"\nSnapped mixed task set (hyperperiod={hyper:.1f}ms):")
    for t in snapped:
        print(f"  {t['name']} prio={t['priority']} model={t['model']} U_actual={t['util_actual']:.3f} T={t['period_ms']:.2f}ms")

