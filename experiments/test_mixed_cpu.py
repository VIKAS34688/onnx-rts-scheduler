"""
Comprehensive CPU test for Task 1 (mixed models) and Task 2 (Baseline SP CPU fallback).
Uses two small stand-in ONNX models (standin_a.onnx and standin_b.onnx).

Runs completely on CPU (RTS_DEVICE=cpu, CUDA_VISIBLE_DEVICES="-1").
Does not touch experiments/results/, docs/phase2/overnight_logs/, or models/profile_cache*.json.
"""

import _common  # noqa: F401
import csv
import os
import random
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

import onnxruntime as ort
import schedulability as sch
import splitter
from _common import make_sample
from generate_taskset_mixed import generate_mixed_task_set, snap_periods_mixed
from run_baseline_mixed import run_baseline_mixed
from run_scheduled_mixed import run_scheduled_mixed
from run_baseline_sp import run_baseline_sp
from run_experiment_mixed import optimize_mixed_task_set, pick_candidates

MODEL_A = "models/standin_a.onnx"
MODEL_B = "models/standin_b.onnx"
TEST_CACHE = "models/standin_cache.json"
TEST_CSV = "test_results_mixed.csv"


class TinyNetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(8 * 16 * 16, 2)

    def forward(self, x):
        return self.fc(self.relu2(self.conv2(self.relu1(self.conv1(x)))).flatten(1))


class TinyNetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(12 * 16 * 16, 2)

    def forward(self, x):
        return self.fc(self.relu2(self.conv2(self.relu1(self.conv1(x)))).flatten(1))


def ensure_standin_models():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_A):
        m_a = TinyNetA().eval()
        x = torch.randn(1, 3, 16, 16)
        torch.onnx.export(m_a, x, MODEL_A, input_names=["input"], output_names=["output"], opset_version=17)
        print(f"[TEST SETUP] Exported {MODEL_A}")
    if not os.path.exists(MODEL_B):
        m_b = TinyNetB().eval()
        x = torch.randn(1, 3, 16, 16)
        torch.onnx.export(m_b, x, MODEL_B, input_names=["input"], output_names=["output"], opset_version=17)
        print(f"[TEST SETUP] Exported {MODEL_B}")


def run_all_tests():
    print("==================================================")
    print("STARTING CPU TESTS FOR MIXED-MODEL & BASELINE SP")
    print("==================================================")
    ensure_standin_models()

    # 1. Check legal cut points
    cuts_a = splitter.list_candidate_split_points(MODEL_A)
    cuts_b = splitter.list_candidate_split_points(MODEL_B)
    print(f"[1] Cuts for model A: {len(cuts_a)}, model B: {len(cuts_b)}")
    assert len(cuts_a) >= 2, f"Model A needs at least 2 cuts, got {len(cuts_a)}"
    assert len(cuts_b) >= 2, f"Model B needs at least 2 cuts, got {len(cuts_b)}"

    # 2. Profile models into dedicated test cache
    samples = {
        MODEL_A: make_sample(MODEL_A),
        MODEL_B: make_sample(MODEL_B),
    }
    c_whole = {
        MODEL_A: sch._profile_config(MODEL_A, (), samples[MODEL_A], sch.PROFILE_CACHE, "models/chunks_test_opt")[0],
        MODEL_B: sch._profile_config(MODEL_B, (), samples[MODEL_B], sch.PROFILE_CACHE, "models/chunks_test_opt")[0],
    }
    sch.save_profile_cache(TEST_CACHE)
    print(f"[2] Profiled whole-model WCETs: A={c_whole[MODEL_A]:.3f}ms, B={c_whole[MODEL_B]:.3f}ms")

    # 3. Generate mixed task set
    rng = random.Random(42)
    models = [MODEL_A, MODEL_B]
    tasks = generate_mixed_task_set(4, 0.7, models, c_whole, rng)
    print(f"[3] Generated {len(tasks)} tasks:")
    for t in tasks:
        print(f"    {t['name']} prio={t['priority']} model={os.path.basename(t['model'])} U={t['util']:.3f} T={t['period_ms']:.2f}ms")
    assigned_models = set(t["model"] for t in tasks)
    assert len(assigned_models) > 0, "Tasks must have assigned models"

    # 4. Snap periods
    snapped, hyper = snap_periods_mixed(tasks, c_whole)
    print(f"[4] Snapped periods (hyperperiod={hyper:.1f}ms):")
    for t in snapped:
        print(f"    {t['name']} T={t['period_ms']:.2f}ms U_act={t['util_actual']:.3f}")
    assert hyper > 0, "Hyperperiod must be positive"

    # 5. Split optimization
    cands_by_model = {
        MODEL_A: pick_candidates(MODEL_A, k=3),
        MODEL_B: pick_candidates(MODEL_B, k=3),
    }
    rts, ok = optimize_mixed_task_set(snapped, cands_by_model, samples, c_whole, method="heuristic",
                                      tmp_dir="models/chunks_test_opt")
    print(f"[5] Split optimization (U=0.7) schedulable: {ok}")

    # Test with a low-utilization set guaranteed to be schedulable
    low_tasks = generate_mixed_task_set(2, 0.2, models, c_whole, random.Random(1))
    low_snapped, _ = snap_periods_mixed(low_tasks, c_whole)
    low_rts, low_ok = optimize_mixed_task_set(low_snapped, cands_by_model, samples, c_whole, method="heuristic",
                                              tmp_dir="models/chunks_test_opt")
    print(f"    Low-utilization set (U=0.2) schedulable: {low_ok}")
    assert low_ok, "Low-utilization task set should be schedulable"

    # 6. Run Baseline Mixed
    print("[6] Running run_baseline_mixed for 1.0s...")
    base_res = run_baseline_mixed(snapped, samples, duration_s=1.0)
    print(f"    Baseline jobs: {base_res['jobs']}, misses: {base_res['misses']}, worst: {base_res['worst_response_ms']:.2f}ms")
    assert base_res["jobs"] > 0, "Baseline should execute at least one job"

    # 7. Run Scheduled Mixed on guaranteed schedulable set
    print("[7] Running run_scheduled_mixed for 1.0s on schedulable set...")
    sched_res = run_scheduled_mixed(low_snapped, low_rts, cands_by_model, samples, duration_s=1.0)
    print(f"    Scheduled jobs: {sched_res['jobs']}, misses: {sched_res['misses']}, "
          f"rule1_ok: {sched_res['rule1_ok']}, preemptions: {sched_res['preemptions']}")
    assert sched_res["jobs"] > 0, "Scheduled should execute at least one job"
    assert sched_res["rule1_ok"], "Rule 1 (no overlap) must be satisfied"

    # 8. Run Baseline SP on CPU
    print("[8] Running run_baseline_sp on CPU for 1.0s...")
    sp_res = run_baseline_sp(snapped, duration_s=1.0)
    print(f"    Baseline SP jobs: {sp_res['jobs']}, misses: {sp_res['misses']}, worst: {sp_res['worst_response_ms']:.2f}ms")
    assert sp_res["jobs"] > 0, "Baseline SP should execute at least one job"

    # 9. Test CLI of run_experiment_mixed.py
    print("[9] Running CLI experiment with run_experiment_mixed.py...")
    cmd = [
        sys.executable, "experiments/run_experiment_mixed.py",
        "--models", MODEL_A, MODEL_B,
        "--max-candidates", "3",
        "--sets", "1",
        "--duration", "0.5",
        "--n-tasks", "2",
        "--utils", "0.6",
        "--snap",
        "--skip-speed-check",
        "--cache-file", TEST_CACHE,
        "--out", TEST_CSV
    ]
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, RTS_DEVICE="cpu", CUDA_VISIBLE_DEVICES="-1"))
    print("CLI Output:")
    print(proc.stdout)
    if proc.returncode != 0:
        print("CLI Stderr:")
        print(proc.stderr)
    assert proc.returncode == 0, f"CLI experiment exited with code {proc.returncode}"
    assert os.path.exists(TEST_CSV), f"{TEST_CSV} was not created"

    # 10. Verify CSV columns
    with open(TEST_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[10] CSV verified: {len(rows)} row(s) found.")
    assert len(rows) > 0, "CSV has no rows"
    assert "models_in_set" in rows[0], "Column 'models_in_set' missing from CSV"
    print(f"    Row 0 models_in_set: {rows[0]['models_in_set']}")
    print(f"    Row 0 analysis_ok: {rows[0]['analysis_ok']}")

    print("\nALL CPU TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
