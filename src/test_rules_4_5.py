"""
Deterministic test of Rule 4 (chunks run in order, output feeding input) and
Rule 5 (a job yields the device at a chunk boundary when a higher-priority job
is waiting), on real split chunks.

Setup: a low-priority job with 3 chunks is started; while its FIRST chunk is
still running (that chunk is artificially slowed by 40 ms so the timing is not
a race), a high-priority single-chunk job is submitted. Expected order:
    low chunk 0 -> high chunk 0 -> low chunk 1 -> low chunk 2
and the final output must equal the whole model's output.

Usage:
    python src/test_rules_4_5.py
"""

import time
import numpy as np
import onnxruntime as ort

from device import DEVICE, PROVIDERS
from scheduler import Scheduler, Task, verify_no_overlap
import splitter

MODEL = "models/alexnet.onnx"


class SlowFirstChunk:
    """Wraps a session; sleeps before running and remembers its last binding."""

    def __init__(self, session, delay_s=0.0):
        self._s, self._delay, self.last_io = session, delay_s, None

    def get_inputs(self):
        return self._s.get_inputs()

    def get_outputs(self):
        return self._s.get_outputs()

    def io_binding(self):
        return self._s.io_binding()

    def run_with_iobinding(self, io):
        time.sleep(self._delay)
        self._s.run_with_iobinding(io)
        self.last_io = io


def main():
    points = splitter.list_candidate_split_points(MODEL)
    n = len(points)
    cuts = [points[n // 3], points[2 * n // 3]]
    paths = splitter.build_chunks(MODEL, cuts, "models/chunks_rule45")
    print(f"[INFO] low-priority job = {len(paths)} chunks, cut after {cuts}")

    low_sessions = [SlowFirstChunk(ort.InferenceSession(p, providers=PROVIDERS),
                                   delay_s=0.04 if i == 0 else 0.0)
                    for i, p in enumerate(paths)]
    whole = ort.InferenceSession(MODEL, providers=PROVIDERS)
    high_session = SlowFirstChunk(whole)

    in_name = whole.get_inputs()[0].name
    x_np = np.random.randn(*[d if isinstance(d, int) else 1 for d in whole.get_inputs()[0].shape]).astype(np.float32)
    x = ort.OrtValue.ortvalue_from_numpy(x_np, DEVICE, 0)
    feed = lambda chunk_index, prev: {in_name: x}

    expected = whole.run(None, {in_name: x_np})[0]

    sched = Scheduler()
    sched.start()
    sched.submit_job(Task("low", 5, low_sessions, feed))
    time.sleep(0.01)                       # low's slow first chunk is now running
    sched.submit_job(Task("high", 1, [high_session], feed))
    time.sleep(1.0)
    sched.stop()

    order = [(e["task_id"], e["chunk_index"]) for e in sorted(sched.execution_log, key=lambda e: e["start"])]
    print(f"[INFO] execution order: {order}")
    want = [("low", 0), ("high", 0), ("low", 1), ("low", 2)]
    print(f"[{'PASS' if order == want else 'FAIL'}] Rule 5: order is {want}")
    print(f"[{'PASS' if sched.preemptions == 1 else 'FAIL'}] preemptions counted: {sched.preemptions} (expected 1)")
    print(f"[{'PASS' if verify_no_overlap(sched.execution_log) else 'FAIL'}] Rule 1: no overlapping chunks")

    got = low_sessions[-1].last_io.get_outputs()[0].numpy()
    diff = float(np.max(np.abs(got - expected)))
    print(f"[{'PASS' if diff < 1e-4 else 'FAIL'}] Rule 4: chunked, preempted job output matches whole model (max diff {diff:.8f})")


if __name__ == "__main__":
    main()