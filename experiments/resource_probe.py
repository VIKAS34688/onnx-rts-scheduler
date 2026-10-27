"""
Paper results R6 and R7, and the question "how many tasks fit on our GPU?":

  1. BASELINE: open 1, 2, 3, ... independent sessions of the model (one per task) and
     record GPU memory and start-up time, until it no longer fits
  2. PROPOSED: open the chunk sessions of the fully split model (tasks that share a
     split configuration share these sessions) and record the same two numbers

GPU memory is what the whole GPU reports as used, so the Windows display counts too:
only the DIFFERENCE from the start is meaningful.

Usage (project root, GPU, other GPU programs closed):
    python experiments/resource_probe.py --model models/alexnet.onnx --max-tasks 12
Results are also saved to docs/phase2/resources_<model>.json
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import gc
import json
import os
import time

import numpy as np
import onnxruntime as ort

import splitter
from _common import make_sample
from device import DEVICE, PROVIDERS
from run_experiment import pick_candidates
from run_scheduled import chunk_paths_for

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/alexnet.onnx")
ap.add_argument("--max-tasks", type=int, default=12)
ap.add_argument("--max-candidates", type=int, default=7)
args = ap.parse_args()


def gpu_used_mb():
    if DEVICE != "cuda":
        return float("nan")
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 20


def open_session(path, sample):
    s = ort.InferenceSession(path, providers=PROVIDERS)
    x = ort.OrtValue.ortvalue_from_numpy(sample, DEVICE, 0)
    io = s.io_binding()
    io.bind_ortvalue_input(s.get_inputs()[0].name, x)
    io.bind_output(s.get_outputs()[0].name, DEVICE)
    for _ in range(3):
        s.run_with_iobinding(io)
    return (s, x, io)


sample = make_sample(args.model)
result = {"model": os.path.basename(args.model), "baseline": [], "proposed": None}

print(f"=== BASELINE: one full-model session per task ({os.path.basename(args.model)}) ===")
print(f"{'tasks':>5} {'GPU MB (added)':>15} {'MB per task':>12} {'start-up (s)':>13}")
base0, t0, held = gpu_used_mb(), time.perf_counter(), []
for n in range(1, args.max_tasks + 1):
    try:
        held.append(open_session(args.model, sample))
    except Exception as e:                       # usually an out-of-memory error
        print(f"  session {n} failed: {type(e).__name__}: {str(e)[:100]}")
        result["max_tasks_that_fit"] = n - 1
        break
    mb = gpu_used_mb() - base0
    result["baseline"].append({"tasks": n, "gpu_mb_added": mb, "startup_s": time.perf_counter() - t0})
    print(f"{n:>5} {mb:>15.0f} {mb / n:>12.0f} {time.perf_counter() - t0:>13.2f}")
else:
    result["max_tasks_that_fit"] = args.max_tasks
print(f"-> up to {result['max_tasks_that_fit']} sessions fit" + (" (stopped at --max-tasks)" if len(held) == args.max_tasks else ""))
del held
gc.collect()
time.sleep(1.0)

print("\n=== PROPOSED: chunk sessions of the fully split model (shared by tasks with the same split) ===")
candidates = pick_candidates(args.model, args.max_candidates)
paths = chunk_paths_for(args.model, candidates, tuple(candidates))
base0, t0 = gpu_used_mb(), time.perf_counter()
held = []
for i, p in enumerate(paths):
    s = ort.InferenceSession(p, providers=PROVIDERS)
    shape = [d if isinstance(d, int) else 1 for d in s.get_inputs()[0].shape]
    x = ort.OrtValue.ortvalue_from_numpy(np.random.randn(*shape).astype(np.float32), DEVICE, 0)
    io = s.io_binding()
    io.bind_ortvalue_input(s.get_inputs()[0].name, x)
    io.bind_output(s.get_outputs()[0].name, DEVICE)
    for _ in range(3):
        s.run_with_iobinding(io)
    held.append((s, x, io))
mb = gpu_used_mb() - base0
result["proposed"] = {"chunks": len(paths), "gpu_mb_added": mb, "startup_s": time.perf_counter() - t0}
one = result["baseline"][0]["gpu_mb_added"] if result["baseline"] else float("nan")
print(f"{len(paths)} chunk sessions: {mb:.0f} MB added, start-up {time.perf_counter() - t0:.2f} s")
print(f"one whole-model session was {one:.0f} MB, so the chunked set costs about {mb / one:.2f}x one baseline task"
      if one == one and one else "")
print("\nThe paper's comparison (Table V) is the total memory of a whole run, baseline against proposed;")
print("here: baseline = N x (MB per task), proposed = one chunk set per distinct split configuration.")

os.makedirs("docs/phase2", exist_ok=True)
json.dump(result, open(f"docs/phase2/resources_{result['model'].replace('.onnx', '')}.json", "w"), indent=1)
