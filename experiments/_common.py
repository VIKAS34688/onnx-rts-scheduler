"""
Shared helpers for the experiment scripts. Importing this module first also
puts ../src on the import path, so `import scheduler` etc. work when a script
is started from the project root (python experiments\\run_experiment.py).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import device  # noqa: E402,F401  -- imports torch first on a CUDA machine (Windows DLL registration)

import numpy as np  # noqa: E402
import onnx  # noqa: E402


def sleep_until(t):
    """Sleep until time.perf_counter() reaches t (returns at once if already past)."""
    delay = t - time.perf_counter()
    if delay > 0:
        time.sleep(delay)

def warm_for(seconds, fn):
    """Call fn() repeatedly for `seconds` of wall-clock time (lets the GPU reach full clocks)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn()

def make_sample(model_path):
    """Random float32 input matching the model's declared input shape."""
    model = onnx.load(model_path, load_external_data=False)
    dims = [d.dim_value or 1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    return np.random.randn(*dims).astype(np.float32)


def summarize(records, tasks):
    """
    records: list of (task_name, release_time, finish_time) from time.perf_counter().
    A job misses its deadline when finish - release > deadline. Response times are
    measured from the NOMINAL release time, so release jitter counts against us.
    """
    deadline = {t["name"]: t["deadline_ms"] for t in tasks}
    misses = {t["name"]: 0 for t in tasks}
    worst = 0.0
    t_ref = min((r[1] for r in records), default=0.0)
    late = []   # (task, release offset in s from the first release, response ms, deadline ms)
    for name, release, finish in records:
        resp_ms = (finish - release) * 1000.0
        worst = max(worst, resp_ms)
        if resp_ms > deadline[name]:
            misses[name] += 1
            late.append((name, round(release - t_ref, 3), round(resp_ms, 2), round(deadline[name], 2)))
    late.sort(key=lambda x: x[1])
    return {
        "jobs": len(records),
        "misses": sum(misses.values()),
        "worst_response_ms": worst,
        "misses_per_task": misses,
        "late": late,
    }