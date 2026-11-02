"""
Naive baseline for mixed-model task sets: every task gets its own InferenceSession
and its own thread for its assigned model, and all threads execute concurrently
with no priorities and no chunking (the paper's "Baseline", Section VIII).

Each task dictionary in `tasks` specifies:
  - name: task name (e.g. 't1')
  - priority: task priority
  - period_ms: task period
  - deadline_ms: task deadline
  - model: path to the ONNX model assigned to this task

Inputs are device-resident and outputs stay on the device (matching run_baseline.py).
"""

import _common  # noqa: F401  -- sets up sys.path for src/
import gc
import math
import os
import threading
import time

import numpy as np
import onnxruntime as ort

from _common import make_sample, sleep_until, summarize, warm_for
from device import DEVICE, PROVIDERS


def run_baseline_mixed(tasks, samples_by_model=None, duration_s=8.0, warmup_runs=20, gc_off=False):
    """
    Execute naive concurrent baseline for a task set with mixed models.
    """
    if samples_by_model is None:
        samples_by_model = {}

    workers = []
    for t in tasks:
        model_path = t["model"]
        if model_path not in samples_by_model:
            samples_by_model[model_path] = make_sample(model_path)
        sample = samples_by_model[model_path]

        session = ort.InferenceSession(model_path, providers=PROVIDERS)
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
        io = session.io_binding()
        io.bind_ortvalue_input(in_name, x)
        io.bind_output(out_name, DEVICE)
        for _ in range(warmup_runs):
            session.run_with_iobinding(io)
        workers.append((t, session, io, x))  # keep x alive while bound

    # Keep the device busy right up to the start to stabilize clocks
    warm_for(1.0, lambda: [s.run_with_iobinding(io) for _, s, io, _ in workers])

    records, lock = [], threading.Lock()
    t0 = time.perf_counter() + 0.05  # Common start so all threads synchronize

    def task_loop(t, session, io):
        period = t["period_ms"] / 1000.0
        for k in range(math.ceil(duration_s / period)):
            release = t0 + k * period
            sleep_until(release)
            session.run_with_iobinding(io)
            finish = time.perf_counter()
            with lock:
                records.append((t["name"], release, finish))

    threads = [threading.Thread(target=task_loop, args=(t, s, io)) for t, s, io, _ in workers]
    if gc_off:
        gc.collect()
        gc.freeze()
        gc.disable()
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        if gc_off:
            gc.enable()
            gc.unfreeze()

    result = summarize(records, tasks)
    del workers
    gc.collect()
    return result


if __name__ == "__main__":
    import random
    from generate_taskset import uunifast

    MODELS = ["models/alexnet.onnx"]
    if os.path.exists("models/resnet18.onnx"):
        MODELS.append("models/resnet18.onnx")

    rng = random.Random(42)
    demo_tasks = [
        {"name": "t1", "priority": 1, "period_ms": 20.0, "deadline_ms": 20.0, "model": MODELS[0]},
        {"name": "t2", "priority": 2, "period_ms": 40.0, "deadline_ms": 40.0, "model": MODELS[-1]},
    ]
    print("[INFO] Running demo run_baseline_mixed...")
    res = run_baseline_mixed(demo_tasks, duration_s=2.0)
    print(f"Jobs: {res['jobs']}, Misses: {res['misses']}, Worst: {res['worst_response_ms']:.2f}ms")

