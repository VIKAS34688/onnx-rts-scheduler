"""
Baseline with CUDA Stream Priorities (the paper's "Baseline SP", Section VIII).

Every task gets its own InferenceSession and its own thread, calling the GPU
concurrently. Unlike the naive Baseline, each task runs on a dedicated CUDA stream
whose priority reflects the task's priority (higher-priority tasks receive higher
CUDA stream priorities).

Implementation using ONNX Runtime + PyTorch:
- PyTorch creates a CUDA stream with a specific priority:
      torch.cuda.Stream(device=0, priority=prio)
  In CUDA, stream priorities are integers where lower values mean higher priority
  (e.g., priority_range() typically returns (0, -1), with -1 being higher priority).
- The raw stream pointer (`stream.cuda_stream`) is passed to ONNX Runtime's
  `CUDAExecutionProvider` via the `user_compute_stream` provider option:
      options = {
          "device_id": 0,
          "user_compute_stream": str(stream.cuda_stream)
      }
- Each inference invocation dispatches into the task's prioritized stream.
- When running in CPU mode (e.g. RTS_DEVICE=cpu or CUDA_VISIBLE_DEVICES="-1"),
  CUDA streams cannot be allocated; the script falls back cleanly to CPUExecutionProvider
  and logs an informative notice.
"""

import _common  # noqa: F401  -- sets up sys.path for src/
import gc
import math
import os
import threading
import time

import numpy as np
import onnxruntime as ort
import torch

from _common import make_sample, sleep_until, summarize, warm_for
from device import DEVICE, PROVIDERS


def compute_stream_priority(task_priority, min_prio, max_prio, num_tasks):
    """
    Map task priority (1 = highest, num_tasks = lowest) to CUDA stream priority
    range [greatest_prio, least_prio] (where greatest_prio is typically -1 and
    least_prio is 0).
    """
    if num_tasks <= 1 or min_prio == max_prio:
        return max_prio
    ratio = (task_priority - 1) / (num_tasks - 1)
    # max_prio is highest (e.g. -1), min_prio is lowest (e.g. 0)
    return int(round(max_prio + ratio * (min_prio - max_prio)))


def run_baseline_sp(tasks, model_path=None, duration_s=8.0, warmup_runs=20, gc_off=False):
    """
    tasks: list of task dicts (name, priority, period_ms, deadline_ms, optional model)
    model_path: default ONNX model path if tasks do not specify t['model']
    """
    cuda_available = (DEVICE == "cuda") and torch.cuda.is_available()

    if cuda_available:
        try:
            least_prio, greatest_prio = torch.cuda.Stream.priority_range()
            print(f"[INFO] CUDA stream priority range: least={least_prio}, greatest={greatest_prio}")
        except Exception as e:
            print(f"[WARN] Failed to query priority_range ({e}), defaulting to 0 and -1")
            least_prio, greatest_prio = 0, -1
    else:
        least_prio, greatest_prio = 0, 0
        print("[INFO] CUDA unavailable or disabled: running CPU fallback for Baseline SP")

    n_tasks = len(tasks)
    workers = []
    stream_priority_map = {}

    for t in tasks:
        m = t.get("model", model_path)
        if not m:
            raise ValueError(f"Task {t.get('name')} has no model specified and no default model_path provided")
        sample = make_sample(m)

        stream = None
        if cuda_available:
            prio = compute_stream_priority(t["priority"], least_prio, greatest_prio, n_tasks)
            stream_priority_map[t["name"]] = prio
            stream = torch.cuda.Stream(device=0, priority=prio)
            provider_options = {
                "device_id": 0,
                "user_compute_stream": str(stream.cuda_stream),
            }
            providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
        else:
            stream_priority_map[t["name"]] = 0
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(m, providers=providers)
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
        io = session.io_binding()
        io.bind_ortvalue_input(in_name, x)
        io.bind_output(out_name, DEVICE)

        for _ in range(warmup_runs):
            session.run_with_iobinding(io)
            if stream is not None:
                stream.synchronize()

        workers.append((t, session, io, x, stream))

    # Stabilize device clocks before starting timed execution
    warm_for(1.0, lambda: [s.run_with_iobinding(io) for _, s, io, _, _ in workers])

    records, lock = [], threading.Lock()
    t0 = time.perf_counter() + 0.05

    def task_loop(t, session, io, stream):
        period = t["period_ms"] / 1000.0
        for k in range(math.ceil(duration_s / period)):
            release = t0 + k * period
            sleep_until(release)
            session.run_with_iobinding(io)
            if stream is not None:
                stream.synchronize()
            finish = time.perf_counter()
            with lock:
                records.append((t["name"], release, finish))

    threads = [threading.Thread(target=task_loop, args=(t, s, io, st))
               for t, s, io, _, st in workers]

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
    result["stream_priorities_used"] = cuda_available
    result["stream_priorities_map"] = stream_priority_map
    del workers
    gc.collect()
    return result


if __name__ == "__main__":
    import random
    from generate_taskset import generate_task_set

    MODEL = "models/alexnet.onnx"
    tasks = generate_task_set(4, 0.7, 2.8, random.Random(1))
    print(f"[INFO] Running run_baseline_sp with {len(tasks)} tasks...")
    res = run_baseline_sp(tasks, MODEL, duration_s=2.0)
    print(f"Jobs: {res['jobs']}, Misses: {res['misses']}, Worst: {res['worst_response_ms']:.2f}ms")
    print(f"Stream priority map: {res.get('stream_priorities_map')}")

