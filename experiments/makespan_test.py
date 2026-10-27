"""
Paper result R1 (Figure 1): does running several DNNs concurrently on one GPU beat
running them one after another? Four models, one thread each (concurrent) against
one thread running them in turn (sequential), 500 repetitions, makespan = time until
the LAST model has finished.

Inputs are GPU-resident and outputs stay on the device, so only GPU work is timed.
In the concurrent case the time includes the barrier that starts all threads
together (a few tens of microseconds).

Usage (project root, GPU):
    python experiments/makespan_test.py --models models/alexnet.onnx models/vgg19.onnx \\
        models/resnet18.onnx models/inception_v4.onnx --tag set1
Results are also saved to docs/phase2/makespan_<tag>.json
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import json
import os
import threading
import time

import numpy as np
import onnxruntime as ort

from _common import make_sample
from device import DEVICE, PROVIDERS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--runs", type=int, default=500)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--tag", default="set1")
args = ap.parse_args()

workers = []
for path in args.models:
    s = ort.InferenceSession(path, providers=PROVIDERS)
    x = ort.OrtValue.ortvalue_from_numpy(make_sample(path), DEVICE, 0)
    io = s.io_binding()
    io.bind_ortvalue_input(s.get_inputs()[0].name, x)
    io.bind_output(s.get_outputs()[0].name, DEVICE)
    workers.append((os.path.basename(path), s, io, x))
print(f"[INFO] {len(workers)} models loaded: {[w[0] for w in workers]}")

# --- sequential: one thread, models in turn
seq = []
for r in range(args.warmup + args.runs):
    t0 = time.perf_counter()
    for _, s, io, _ in workers:
        s.run_with_iobinding(io)
    if r >= args.warmup:
        seq.append((time.perf_counter() - t0) * 1000.0)

# --- concurrent: one thread per model, started together by a barrier
total = args.warmup + args.runs
start, end = threading.Barrier(len(workers) + 1), threading.Barrier(len(workers) + 1)


def loop(s, io):
    for _ in range(total):
        start.wait()
        s.run_with_iobinding(io)
        end.wait()


threads = [threading.Thread(target=loop, args=(s, io)) for _, s, io, _ in workers]
for th in threads:
    th.start()
con = []
for r in range(total):
    t0 = time.perf_counter()
    start.wait()
    end.wait()
    if r >= args.warmup:
        con.append((time.perf_counter() - t0) * 1000.0)
for th in threads:
    th.join()


def stats(v):
    return {"mean_ms": float(np.mean(v)), "p50_ms": float(np.percentile(v, 50)),
            "p99_ms": float(np.percentile(v, 99)), "max_ms": float(np.max(v))}


res = {"models": [w[0] for w in workers], "runs": args.runs, "sequential": stats(seq), "concurrent": stats(con)}
res["concurrent_over_sequential_mean"] = res["concurrent"]["mean_ms"] / res["sequential"]["mean_ms"]
os.makedirs("docs/phase2", exist_ok=True)
json.dump(res, open(f"docs/phase2/makespan_{args.tag}.json", "w"), indent=1)

print(f"\n{'':<12} {'mean':>9} {'p50':>9} {'p99':>9} {'max':>9}   (makespan in ms, {args.runs} runs)")
for name in ("sequential", "concurrent"):
    r = res[name]
    print(f"{name:<12} {r['mean_ms']:>9.2f} {r['p50_ms']:>9.2f} {r['p99_ms']:>9.2f} {r['max_ms']:>9.2f}")
print(f"\nconcurrent / sequential (mean) = {res['concurrent_over_sequential_mean']:.2f}  "
      "(1.00 = no gain from concurrency; the paper found about 1.0 or worse)")
