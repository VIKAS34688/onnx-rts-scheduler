"""
Scheduled run for mixed-model task sets: every job goes through scheduler.py
(Rules 1-5) and executes as the chunk sequence chosen by the split optimization
for its assigned model.

Inputs and intermediate tensors are device-resident (matching run_scheduled.py).
"""

import _common  # noqa: F401  -- sets up sys.path for src/
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

import hashlib          # add at the top


def chunk_paths_for(model_path, candidates, active_points):
    """Chunk files for one split configuration (built once, then reused)."""
    if not active_points:
        return [model_path]
    stem = os.path.splitext(os.path.basename(model_path))[0]
    tag = hashlib.md5("|".join(map(str, active_points)).encode()).hexdigest()[:10]
    out_dir = os.path.join("models", "chunks_run", stem, f"cfg_{tag}")
    paths = [os.path.join(out_dir, f"chunk_{i}.onnx") for i in range(len(active_points) + 1)]
    if all(os.path.exists(p) for p in paths):
        return paths
    return splitter.build_chunks(model_path, list(active_points), out_dir)

def _warm_up(sessions, x, n):
    """Run the whole chunk chain n times, handing tensors on by name like the scheduler does."""
    for _ in range(n):
        store = {}
        for s in sessions:
            io = s.io_binding()
            for meta in s.get_inputs():
                io.bind_ortvalue_input(meta.name, store.get(meta.name, x))
            out_names = [m.name for m in s.get_outputs()]
            for name in out_names:
                io.bind_output(name, DEVICE)
            s.run_with_iobinding(io)
            for name, value in zip(out_names, io.get_outputs()):
                store[name] = value


def run_scheduled_mixed(tasks, rts_tasks, candidates_by_model, samples_by_model=None,
                        duration_s=8.0, warmup_runs=5, gc_off=False):
    """
    tasks:                task dicts (each containing 'name', 'priority', 'period_ms', 'model')
    rts_tasks:            the optimized RTSTask objects (each with .active_points)
    candidates_by_model:  dict mapping model_path -> list of candidate split points
    samples_by_model:     dict mapping model_path -> sample numpy array
    """
    if samples_by_model is None:
        samples_by_model = {}

    rts_by_name = {r.name: r for r in rts_tasks}
    sessions_by_cfg, plan = {}, []

    for t in tasks:
        m = t["model"]
        if m not in samples_by_model:
            samples_by_model[m] = make_sample(m)
        sample = samples_by_model[m]
        candidates = candidates_by_model[m]
        points = tuple(rts_by_name[t["name"]].active_points)

        cfg_key = (m, points)
        if cfg_key not in sessions_by_cfg:
            paths = chunk_paths_for(m, candidates, points)
            sessions_by_cfg[cfg_key] = [ort.InferenceSession(p, providers=PROVIDERS) for p in paths]
        sessions = sessions_by_cfg[cfg_key]
        x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
        _warm_up(sessions, x, warmup_runs)
        plan.append((t, sessions, x))

    # Keep the device busy right up to the start to stabilize clocks
    warm_for(1.0, lambda: [_warm_up(sessions, x, 1) for _, sessions, x in plan])

    if gc_off:
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

    deadline = time.perf_counter() + 30.0  # Drain queue
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

