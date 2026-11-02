# Phase 2 Agent Implementation Notes

This document details the work implemented during the overnight run under strict CPU isolation rules (`$env:CUDA_VISIBLE_DEVICES="-1"`). No GPU sessions were initialized, and all existing files, caches, and logs remained untouched.

Git branch: `phase2-agent`

---

## Task 1: Mixed-Model Task Sets (`run_experiment_mixed.py`)

### What Was Built
1. **`experiments/generate_taskset_mixed.py`**:
   - `generate_mixed_task_set(n, total_u, models, c_periods_by_model, rng)`: Implements the paper's multi-model protocol (Section VIII-A). Each task uniformly samples a model from `--models`. Task utilizations are generated via UUniFast ($U_i \ge 0.03$). Each task's period is computed from the profiled whole-model WCET of its own assigned model ($T_i = D_i = C_i^{\text{whole}} / U_i$). Tasks are assigned Deadline-Monotonic priorities ($1 = \text{highest}$).
   - `snap_periods_mixed(tasks, c_periods_by_model)`: Computes a unified base unit in `NICE_UNITS` fitting all models, snaps task periods to allowed multiples of the base unit ($\ge 1.05 C_i$), and calculates the exact task set hyperperiod using integer LCM (`math.lcm`).

2. **`experiments/run_baseline_mixed.py`**:
   - `run_baseline_mixed(tasks, samples_by_model, duration_s, warmup_runs, gc_off)`: Naive concurrent execution of mixed models. Each task instantiates its own `InferenceSession` for its assigned model, binds GPU/device inputs and outputs via IOBinding, and executes concurrently across threads with no priorities or chunking.

3. **`experiments/run_scheduled_mixed.py`**:
   - `run_scheduled_mixed(tasks, rts_tasks, candidates_by_model, samples_by_model, duration_s, warmup_runs, gc_off)`: Dispatches mixed-model tasks through `scheduler.py` (Rules 1–5). Sessions are cached per `(model_path, active_points)` configuration so tasks sharing a model and cut configuration share sessions. Device-resident intermediate tensors are handed over between chunks.

4. **`experiments/run_experiment_mixed.py`**:
   - End-to-end experiment driver for mixed task sets.
   - Supports candidate selection per model via `pick_candidates`.
   - Supports `--method heuristic` (Algorithm 2) and `--method optimal` (Algorithm 1).
   - Writes all standard CSV columns from `run_experiment.py` plus the new `models_in_set` column (formatted as `model_1+model_2+...` in priority order).

5. **`experiments/test_mixed_cpu.py`**:
   - Automated 10-step verification suite that exports two stand-in models (`models/standin_a.onnx`, `models/standin_b.onnx`) and tests cut selection, task generation, period snapping, split optimization, baseline execution, scheduled execution, Baseline SP, and end-to-end CLI execution.

### Design Choices & Simplifications
- **Base Unit for Mixed Snapping**: A common base unit is selected from `NICE_UNITS` such that `base_unit <= min(C_i)/3` and `120 * base_unit >= 1.05 * max(C_i)`. This guarantees that every task has valid multiples in `PERIOD_MULTIPLES` (all dividing 120), ensuring a bounded and exact hyperperiod.
- **`models_in_set` Column Format**: Formatted as `model1+model2+...` (using `os.path.basename`) to avoid comma collision inside unquoted CSV columns.
- **`model` Column in Mixed CSV**: Recorded as `"mixed"`, distinguishing mixed-model experiments from single-model sweeps while `models_in_set` details the exact assignment per task.

### Tested on CPU
- Legal cut points extraction on multi-model pipelines (`splitter.list_candidate_split_points`).
- Mixed task generation and snapping with exact LCM hyperperiod computation.
- Greedy and optimal split optimization across tasks with different models and candidate cuts.
- Baseline concurrent multi-threading with mixed model inference sessions.
- Scheduled dispatcher execution verifying Rule 1 (`rule1_ok = True`).
- Full CLI experiment execution with `--snap` and CSV output verification.

### NOT Tested on CPU (Unverified)
- GPU-side execution and speed checks against NVIDIA CUDA drivers.
- Inter-model GPU cache and memory contention on the GTX 1650.

### Exact Morning GPU Commands
```powershell
# 1. Quick test (2 sets, duration 4s, AlexNet + ResNet-18)
python experiments/run_experiment_mixed.py --models models/alexnet.onnx models/resnet18.onnx --sets 2 --duration 4.0 --snap --out experiments/results/mixed_quick_test.csv

# 2. Full paper sweep (AlexNet + ResNet-18 + VGG-19, 10 sets per utilization, heuristic)
python experiments/run_experiment_mixed.py --models models/alexnet.onnx models/resnet18.onnx models/vgg19.onnx --n-tasks 4 --utils 0.6 0.7 0.8 0.9 --sets 10 --snap --duration 8.0 --method heuristic --max-candidates 7 --overhead-ms 0.35 --out experiments/results/mixed_sweep_h7.csv

# 3. Optimal comparison (5 candidate cuts, Algorithm 1)
python experiments/run_experiment_mixed.py --models models/alexnet.onnx models/resnet18.onnx --n-tasks 3 --utils 0.6 0.7 0.8 0.9 --sets 5 --snap --duration 8.0 --method optimal --max-candidates 5 --overhead-ms 0.35 --out experiments/results/mixed_sweep_o5.csv
```

---

## Task 2: Baseline SP (`run_baseline_sp.py`)

### What Was Built
1. **`experiments/run_baseline_sp.py`**:
   - Replicates the paper's "Baseline SP" (Section VIII: concurrent execution where tasks run on prioritized CUDA streams).
   - Queries `torch.cuda.Stream.priority_range()` (on CUDA, `leastPriority = 0`, `greatestPriority = -1`).
   - Maps Deadline-Monotonic task priority $1..N$ to stream priority values.
   - Creates a dedicated PyTorch CUDA stream: `torch.cuda.Stream(device=0, priority=prio)`.
   - Obtains the native stream handle `stream.cuda_stream` (integer memory pointer) and configures ONNX Runtime's `CUDAExecutionProvider`:
     ```python
     provider_options = {
         "device_id": 0,
         "user_compute_stream": str(stream.cuda_stream),
     }
     providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
     session = ort.InferenceSession(model_path, providers=providers)
     ```
   - Synchronizes each task's stream (`stream.synchronize()`) after `session.run_with_iobinding(io)` so response times measure true GPU kernel completion without stalling other concurrent streams.
   - Provides an automated CPU fallback for debugging and non-CUDA environments.

### Investigation of `user_compute_stream` on `onnxruntime-gpu 1.20.2`
- **Binary Inspection**: Searched strings inside `venv/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll`. Confirmed that `user_compute_stream` and `has_user_compute_stream` are first-class compiled provider options in `CUDAExecutionProviderInfo::FromProviderOptions`.
- **Option Validation**: Tested passing `{'user_compute_stream': str(ptr)}` versus an invalid option `{'totally_invalid_option_xyz': '1'}`. ONNX Runtime immediately rejected the invalid option with `[ONNXRuntimeError] Unknown provider option`, but successfully recognized and parsed `user_compute_stream`.
- **Reference Code**: Confirmed that official ONNX Runtime transformers utilities (`transformers/io_binding_helper.py`, line 342) also utilize `options["user_compute_stream"] = str(stream)`.
- **Preconditions**:
  1. `import torch` must precede ONNX Runtime CUDA initialization on Windows so bundled CUDA DLLs are registered.
  2. `user_compute_stream` is mutually exclusive with `gpu_external_alloc` (an external allocator), which our project does not use.

### Tested on CPU
- Provider option parsing in `onnxruntime_providers_cuda.dll`.
- PyTorch `torch.cuda.Stream` API inspection (verifying presence of `cuda_stream`, `priority`, `priority_range`).
- Fallback CPU multi-threaded execution path in `run_baseline_sp.py`.

### NOT Tested on CPU (Unverified)
- True GPU kernel execution on prioritized CUDA streams (GTX 1650 was reserved for overnight runs).
- Hardware scheduling preemption behavior between priority -1 and priority 0 streams on Turing architecture.

### Exact Morning GPU Commands
```powershell
# 1. Quick demo on AlexNet
python experiments/run_baseline_sp.py

# 2. Compare naive Baseline vs Baseline SP on the same task set
python -c "import _common, random; from generate_taskset import generate_task_set; from run_baseline import run_baseline; from run_baseline_sp import run_baseline_sp; tasks = generate_task_set(4, 0.7, 2.8, random.Random(1)); print('Naive Baseline:', run_baseline(tasks, 'models/alexnet.onnx', duration_s=5.0)); print('Baseline SP:', run_baseline_sp(tasks, 'models/alexnet.onnx', duration_s=5.0))"
```

---

## Compilation and File Inventory
All new Python files compiled cleanly with `python -m py_compile`:
- `experiments/generate_taskset_mixed.py`
- `experiments/run_baseline_mixed.py`
- `experiments/run_scheduled_mixed.py`
- `experiments/run_experiment_mixed.py`
- `experiments/run_baseline_sp.py`
- `experiments/test_mixed_cpu.py`

