"""
Measures what the Python dispatcher itself costs per chunk, which the response-time
analysis does not know about (profiler.py times only run_with_iobinding()).

All jobs are queued first, then the dispatcher is started, so it runs back to back.
The gap between the end of one chunk and the start of the next one is then pure
dispatch overhead: popping the queue, creating and binding the IOBinding, the
Rule-5 check, and the completion callback.

Usage (clean conditions: charger in, apps closed):
    python experiments/measure_dispatch_overhead.py
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import numpy as np
import onnxruntime as ort

import splitter
from _common import make_sample, warm_for
from device import DEVICE, PROVIDERS
from run_experiment import MODEL, CANDIDATE_IDX
from run_scheduled import chunk_paths_for
from scheduler import Scheduler, Task

N_JOBS = 400

candidates = [splitter.list_candidate_split_points(MODEL)[i] for i in CANDIDATE_IDX]
sample = make_sample(MODEL)
x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)

configs = {"1 chunk": (), "2 chunks": (candidates[4],),
           "4 chunks": (candidates[1], candidates[3], candidates[5])}

print(f"{'config':>9} {'gaps':>6} {'p50 (ms)':>9} {'p95 (ms)':>9} {'p99 (ms)':>9} {'max (ms)':>9}")
worst_p99 = 0.0
for label, points in configs.items():
    sessions = [ort.InferenceSession(p, providers=PROVIDERS)
                for p in chunk_paths_for(MODEL, candidates, points)]
    in_name = sessions[0].get_inputs()[0].name
    feed = lambda chunk_index, prev: {in_name: x}

    def one_pass():
        current = x
        for s in sessions:
            io = s.io_binding()
            io.bind_ortvalue_input(s.get_inputs()[0].name, current)
            io.bind_output(s.get_outputs()[0].name, DEVICE)
            s.run_with_iobinding(io)
            current = io.get_outputs()[0]

    warm_for(1.0, one_pass)

    sched = Scheduler()
    for _ in range(N_JOBS):
        sched.submit_job(Task("job", 1, sessions, feed))
    sched.start()
    sched.stop()                     # returns once the queue is empty

    log = sorted(sched.execution_log, key=lambda e: e["start"])
    gaps = np.array([(log[i + 1]["start"] - log[i]["end"]) * 1000.0 for i in range(len(log) - 1)])
    p50, p95, p99 = (float(np.percentile(gaps, q)) for q in (50, 95, 99))
    worst_p99 = max(worst_p99, p99)
    print(f"{label:>9} {len(gaps):>6} {p50:>9.4f} {p95:>9.4f} {p99:>9.4f} {gaps.max():>9.4f}")

print(f"\nlargest p99 gap = {worst_p99:.4f} ms per chunk dispatch")
print("suggested value for --overhead-ms (rounded up to 0.05):",
      f"{np.ceil(worst_p99 / 0.05) * 0.05:.2f}")