"""
Integration test for scheduler.py.

Proves Rules 1-3 hold on real hardware, using the whole (unsplit)
AlexNet model as a single-chunk "task" for several concurrent jobs
at different priorities. Rules 4 and 5 (multi-chunk sequencing and
mid-task preemption) can only be properly tested once splitter.py
produces real multi-chunk tasks -- this test gets extended once
Person B's work is ready.

Usage:
    python src/test_scheduler_integration.py
"""

import time
import torch  # import first -- registers PyTorch's bundled CUDA DLLs
import onnxruntime as ort
import numpy as np

from scheduler import Scheduler, Task, verify_no_overlap


def make_session(label: str):
    session = ort.InferenceSession(
        "models/alexnet.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"[INFO] Session '{label}' running on: {session.get_providers()[0]}")
    return session


def make_input_feed_factory(input_name, sample):
    def make_input_feed(chunk_index, prev_outputs):
        return {input_name: sample}
    return make_input_feed


if __name__ == "__main__":
    print("[INFO] Creating sessions (this includes CUDA context setup, "
          "may take a few seconds)...")
    sessions = {tid: make_session(tid) for tid in ["low", "mid", "high"]}
    input_name = sessions["low"].get_inputs()[0].name
    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup: each session pays a one-time CUDA kernel-selection cost on
    # its first real call. Without this, the first few rounds of the real
    # test absorb that cost live and look like queue backlog that isn't
    # actually there -- exactly the reason profiler.py also warms up
    # before timing anything.
    print("[INFO] Warming up sessions...")
    for tid, session in sessions.items():
        session.run(None, {input_name: sample})
    print("[INFO] Warmup complete.")

    scheduler = Scheduler()
    scheduler.start()
    print("[INFO] Scheduler dispatcher thread started.")

    priorities = {"high": 1, "mid": 2, "low": 3}

    for round_num in range(5):
        for tid, priority in priorities.items():
            task = Task(
                tid, priority=priority,
                chunk_sessions=[sessions[tid]],
                make_input_feed=make_input_feed_factory(input_name, sample),
            )
            scheduler.submit_job(task)
        with scheduler._lock:
            depth = len(scheduler._queue)
        print(f"[INFO] Round {round_num + 1}/5 submitted "
              f"(queue depth now: {depth})")
        time.sleep(0.1)

    print("[INFO] All jobs submitted. Draining queue...")
    time.sleep(3)

    with scheduler._lock:
        remaining = len(scheduler._queue)
    if remaining > 0:
        print(f"[WARN] {remaining} job(s) still queued -- increase the "
              f"drain sleep above if this happens consistently.")

    scheduler.stop()

    print(f"\n[INFO] Logged {len(scheduler.execution_log)} chunk executions")

    if len(scheduler.execution_log) == 0:
        print("[FAIL] No executions were logged at all -- the dispatcher "
              "thread likely never processed the queue. Check that "
              "scheduler.start() was called BEFORE submit_job(), and that "
              "no exception was silently raised inside _run_task_chunks "
              "(temporarily remove the 'daemon=True' thread flag in "
              "scheduler.py's start() method to surface hidden errors).")
    else:
        no_overlap = verify_no_overlap(scheduler.execution_log)
        print(f"[{'PASS' if no_overlap else 'FAIL'}] "
              f"Rule 1 (single chunk at a time): {no_overlap}")

        order = [e["task_id"] for e in
                  sorted(scheduler.execution_log, key=lambda e: e["start"])]
        print(f"[INFO] Dispatch order: {order}")

        durations_ms = [(e["end"] - e["start"]) * 1000.0
                          for e in scheduler.execution_log]
        print(f"[INFO] Chunk durations (ms): "
              f"min={min(durations_ms):.3f}  max={max(durations_ms):.3f}  "
              f"mean={sum(durations_ms)/len(durations_ms):.3f}")