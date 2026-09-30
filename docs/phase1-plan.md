# Phase 1 plan (Sept 30 - Oct 15)

Goal: validate the engine swap (TensorRT -> ONNX Runtime) on a classifier
(AlexNet) before touching object detection in Phase 2.

- Days 1-3: environment setup, confirm CUDA + stream priority levels on
  the GTX 1650, export AlexNet to ONNX, re-derive Eq. (1)-(4) as a team
- Days 4-10: parallel build
  - Person A: scheduler.py + profiler.py (single-chunk enforcement,
    PCIe overhead measurement)
  - Person B: splitter.py + schedulability.py (Algorithm 1/2 loop)
  - Person C: generate_taskset.py + run_baseline.py / run_scheduled.py
- Days 11-12: integration
- Day 13: run the small-scale schedulability experiment
- Days 14-15: write-up, demo prep, submit Oct 15

Deliverables: working split-and-schedule pipeline on AlexNet, empirically
confirmed single-chunk GPU execution, measured PCIe overhead, small-scale
schedulability comparison (scheduled vs. baseline), written disclosure of
hardware differences from the paper's Jetson setup.
