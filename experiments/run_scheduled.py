"""
Scheduled run: the same periodic task set, but every job goes through
scheduler.py (Rules 1-5) and is executed as the chunk sequence chosen by the
split optimization (each task has its own split configuration).

Chunk files are built once per configuration and reused (models/chunks_run/).
Inputs are GPU-resident and outputs stay on the device, same as run_baseline.py.

Usage: normally called from run_experiment.py.
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import gc
import math
import os
import threading
import time

import onnxruntime as ort

from _common import make_sample, sleep_until, summarize, warm_for
from device import DEVICE, PROVIDERS
from scheduler import Scheduler, Task, verify_no_overlap
import splitter


def chunk_paths_for(model_path, candidates, active_points):
    """Chunk files for one split configuration (built once, then reused)."""
    if not active_points:
        return [model_path]
    stem = os.path.splitext(os.path.basename(model_path))[0]
    tag = "_".join(str(candidates.index(p)) for p in active_points)
    out_dir = os.path.join("models", "chunks_run", stem, f"cfg_{tag}")
    paths = [os.path.join(out_dir, f"chunk_{i}.onnx") for i in range(len(active_points) + 1)]
    if all(os.path.exists(p) for p in paths):
        return paths
    return splitter.build_chunks(model_path, list(active_points), out_dir)


def _warm_up(sessions, x, n):
    for _ in range(n):
        current = x
        for s in sessions:
            io = s.io_binding()
            io.bind_ortvalue_input(s.get_inputs()[0].name, current)
            io.bind_output(s.get_outputs()[0].name, DEVICE)
            s.run_with_iobinding(io)
            current = io.get_outputs()[0]


def run_scheduled(tasks, rts_tasks, model_path, candidates, duration_s=8.0, warmup_runs=5,
                  gc_off=False):
    """
    tasks:     task dicts from generate_task_set
    rts_tasks: the optimized RTSTask objects (each has .active_points), same names
    """
    sample = make_sample(model_path)
    rts_by_name = {r.name: r for r in rts_tasks}

    sessions_by_cfg, plan = {}, []
    for t in tasks:
        points = tuple(rts_by_name[t["name"]].active_points)
        if points not in sessions_by_cfg:
            paths = chunk_paths_for(model_path, candidates, points)
            sessions_by_cfg[points] = [ort.InferenceSession(p, providers=PROVIDERS) for p in paths]
        sessions = sessions_by_cfg[points]
        x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
        _warm_up(sessions, x, warmup_runs)
        plan.append((t, sessions, x))

    # keep the GPU busy right up to the start, otherwise its clocks drop during the idle gap
    warm_for(1.0, lambda: [_warm_up(sessions, x, 1) for _, sessions, x in plan])

    if gc_off:                      # no Python garbage-collection pauses during the timed region
        gc.collect()
        gc.freeze()
        gc.disable()
    scheduler = Scheduler()
    scheduler.start()
    records, lock = [], threading.Lock()
    total_jobs = sum(math.ceil(duration_s / (t["period_ms"] / 1000.0)) for t in tasks)
    t0 = time.perf_counter() + 0.05

    def make_callback(name, release):
        def done(finish):
            with lock:
                records.append((name, release, finish))
        return done

    def release_loop(t, sessions, x):
        period = t["period_ms"] / 1000.0
        in_name = sessions[0].get_inputs()[0].name
        feed = lambda chunk_index, prev: {in_name: x}
        for k in range(math.ceil(duration_s / period)):
            release = t0 + k * period
            sleep_until(release)
            scheduler.submit_job(Task(t["name"], t["priority"], sessions, feed,
                                      on_complete=make_callback(t["name"], release)))

    threads = [threading.Thread(target=release_loop, args=p) for p in plan]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    deadline = time.perf_counter() + 30.0          # let the queue drain
    while time.perf_counter() < deadline:
        with lock:
            if len(records) >= total_jobs:
                break
        time.sleep(0.05)
    scheduler.stop()
    if gc_off:
        gc.enable()
        gc.unfreeze()

    result = summarize(records, tasks)
    result["unfinished"] = total_jobs - len(records)
    result["rule1_ok"] = verify_no_overlap(scheduler.execution_log)
    result["preemptions"] = scheduler.preemptions
    del sessions_by_cfg, plan
    gc.collect()
    return result