# RTS DNN Multitasking — ONNX Runtime Extension

Real-Time Systems course project (IIIT Dharwad), extending:

> Aromolo, Stevanato, Biondi, Buttazzo — *"Real-Time Multitasking of Deep Neural
> Networks With Nvidia TensorRT,"* RTSS 2025.

## What this project does

The reference paper builds a priority-based, chunk-splitting scheduler for
running multiple DNNs on one shared GPU with predictable timing, using
NVIDIA TensorRT on Jetson SoCs (unified CPU-GPU memory), validated only on
image classifiers.

This project ports the same scheduling idea along two axes:

1. **Engine swap** — TensorRT → **ONNX Runtime**, on a **discrete GPU**
   (GTX 1650) instead of a unified-memory Jetson SoC.
2. **Domain swap** — image classification → **object detection**
   (YOLO-nano / MobileNet-SSD), splitting only the sequential backbone and
   leaving the branching neck/heads as one unsplit chunk.

## Phase 1 scope (this repo, current milestone)

Validate the engine swap on a classifier first, before touching detection,
to isolate the engine-swap effect from the domain-swap effect.

- Model: **AlexNet** (fully sequential, no skip connections — the safest
  model to validate graph-splitting against)
- Reuse the reference paper's Equations (1)-(4) and Algorithms 1/2 as-is —
  no new scheduling theory, this is a replication-and-port study
- Empirically verify "one chunk on the GPU at a time" (Rule 1) rather than
  assuming it
- Measure, not guess, the PCIe copy overhead introduced by the discrete GPU

## Repo structure

```
src/
  splitter.py        — ONNX graph splitting (onnx.utils.extract_model wrapper)
  profiler.py         — per-chunk WCET profiling (CUDA event + wall-clock timing)
  schedulability.py   — Eq. (1)-(4) response-time analysis + Algorithm 1/2
  scheduler.py         — runtime dispatcher enforcing Rules 1-5
experiments/
  generate_taskset.py — UUniFast task-set generation
  run_baseline.py      — naive concurrent inference (no scheduler)
  run_scheduled.py     — scheduled inference using scheduler.py
models/               — exported .onnx files (gitignored — see below)
docs/                  — write-ups, plots, phase reports
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Team

3-member team, RTS course project, Dr. Swagatika Sahoo.

- Person A — runtime scheduler, single-chunk enforcement, PCIe overhead measurement
- Person B — graph splitting, schedulability analysis + optimization algorithms
- Person C — experiment harness, task-set generation, results/plots

## Status

Phase 1 in progress — see `docs/` for the current day-by-day plan.
