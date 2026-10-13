"""
Naive baseline: every task gets its own InferenceSession and its own thread,
and all of them call the GPU concurrently with no priorities and no chunking
(the paper's "Baseline", Section VIII).

Jobs are released periodically at t0 + k*T. If a job is still running when
the next one is released, the next one starts late (a backlog builds up).
Inputs are GPU-resident and outputs stay on the device, the same convention
as profiler.py and scheduler.py, so the comparison is compute vs compute.
(PCIe upload/download is therefore NOT included in either mode.)

Usage (quick demo, run from the project root):
    python experiments/run_baseline.py
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import gc
import math
import threading
import time

import numpy as np
import onnxruntime as ort

from _common import make_sample, sleep_until, summarize
from device import DEVICE, PROVIDERS


def run_baseline(tasks, model_path, duration_s=8.0, warmup_runs=20, gc_off=False):
    sample = make_sample(model_path)

    workers = []
    for t in tasks:
        session = ort.InferenceSession(model_path, providers=PROVIDERS)
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
        io = session.io_binding()
        io.bind_ortvalue_input(in_name, x)
        io.bind_output(out_name, DEVICE)
        for _ in range(warmup_runs):
            session.run_with_iobinding(io)
        workers.append((t, session, io, x))   # keep x alive while bound

    records, lock = [], threading.Lock()
    t0 = time.perf_counter() + 0.5            # common start, so all threads line up

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
    if gc_off:                      # no Python garbage-collection pauses during the timed region
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
    from generate_taskset import generate_task_set

    MODEL = "models/alexnet.onnx"
    tasks = generate_task_set(4, 0.7, 2.8, random.Random(1))
    print(run_baseline(tasks, MODEL, duration_s=5.0))