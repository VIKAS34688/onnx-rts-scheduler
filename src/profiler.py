"""
Per-chunk timing profiling.

The reference paper profiles engine execution times from the CPU side
using high-resolution wall-clock timers, isolating each chunk (running
it alone on the GPU) to get clean WCET estimates uncontaminated by
resource contention. We follow the same wall-clock approach here.

Inputs and outputs are kept on the device with IOBinding, so the numbers are compute time
alone, not PCIe copies (measure_pcie_overhead() reports those separately).

A chunk may have several inputs and several outputs (multi-tensor cuts). Only the first
chunk of a model takes the real sample; every other input is a random float tensor of the
shape the chunk declares. Convolution/Gemm timing does not depend on the values.

Warm-up is time-based as well as count-based: after an idle gap the GPU sits at low clocks.
"""

import time
import numpy as np
import onnxruntime as ort

from device import DEVICE, PROVIDERS

WARMUP_SECONDS = 0.5   # warm up by time, so the GPU reaches full clocks before timing starts


def _percentile_stats(times_ms: list) -> dict:
    arr = np.array(times_ms)
    return {
        "min_ms": float(arr.min()),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "n_runs": len(arr),
    }


def _print_stats(label: str, stats: dict):
    print(f"  {label}: min={stats['min_ms']:.3f}ms  mean={stats['mean_ms']:.3f}ms  "
          f"p50={stats['p50_ms']:.3f}ms  p95={stats['p95_ms']:.3f}ms  "
          f"p99={stats['p99_ms']:.3f}ms  max={stats['max_ms']:.3f}ms  "
          f"(n={stats['n_runs']})")


def _inputs_for_session(session, sample_input: np.ndarray) -> dict:
    """name -> array for every declared input: the sample where it fits, random float32 elsewhere."""
    feed = {}
    for k, meta in enumerate(session.get_inputs()):
        if meta.type != "tensor(float)":
            raise NotImplementedError(f"input {meta.name} has type {meta.type}; only float inputs are supported")
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        if k == 0 and list(sample_input.shape) == shape:
            feed[meta.name] = sample_input
        else:
            feed[meta.name] = np.random.randn(*shape).astype(np.float32)
    return feed


def profile_chunk(chunk_onnx_path: str, sample_input: np.ndarray,
                   n_runs: int = 300, warmup: int = 10) -> dict:
    """
    Run a chunk n_runs times in isolation and return compute-only timing stats
    (min/mean/p50/p95/p99/max, in milliseconds).
    """
    session = ort.InferenceSession(chunk_onnx_path, providers=PROVIDERS)
    feed = _inputs_for_session(session, sample_input)

    io_binding = session.io_binding()
    keep = []                                              # keep the device tensors alive while bound
    for name, arr in feed.items():
        ov = ort.OrtValue.ortvalue_from_numpy(arr, DEVICE, 0)
        keep.append(ov)
        io_binding.bind_ortvalue_input(name, ov)
    for meta in session.get_outputs():
        io_binding.bind_output(meta.name, DEVICE)

    # Warm up by TIME, not run count: after an idle gap the GPU sits at low clocks.
    warm_until = time.perf_counter() + WARMUP_SECONDS
    runs = 0
    while runs < warmup or time.perf_counter() < warm_until:
        session.run_with_iobinding(io_binding)
        runs += 1

    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run_with_iobinding(io_binding)             # compute only, no copy
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return _percentile_stats(times_ms)


def measure_pcie_overhead(chunk_onnx_path: str, sample_input: np.ndarray,
                           n_runs: int = 300, warmup: int = 10) -> dict:
    """
    Upload (host->device) and download (device->host) costs of the FIRST input and FIRST output,
    separately from compute time: the overhead the paper's Jetson setup does not have.
    """
    session = ort.InferenceSession(chunk_onnx_path, providers=PROVIDERS)
    feed = _inputs_for_session(session, sample_input)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    arr = feed[input_name]

    for _ in range(warmup):
        gpu_input = ort.OrtValue.ortvalue_from_numpy(arr, DEVICE, 0)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_name, gpu_input)
        io_binding.bind_output(output_name, DEVICE)
        session.run_with_iobinding(io_binding)
        _ = io_binding.get_outputs()[0].numpy()

    upload_times_ms, download_times_ms = [], []

    # --- upload: host -> device ---
    for _ in range(n_runs):
        # --- upload: host -> device ---
        start = time.perf_counter()
        gpu_input = ort.OrtValue.ortvalue_from_numpy(arr, DEVICE, 0)
        end = time.perf_counter()
        upload_times_ms.append((end - start) * 1000.0)

        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_name, gpu_input)
        io_binding.bind_output(output_name, DEVICE)
        session.run_with_iobinding(io_binding)

        # --- download: device -> host ---
        gpu_output = io_binding.get_outputs()[0]
        start = time.perf_counter()
        _ = gpu_output.numpy()
        end = time.perf_counter()
        download_times_ms.append((end - start) * 1000.0)

    return {"upload": _percentile_stats(upload_times_ms), "download": _percentile_stats(download_times_ms)}


if __name__ == "__main__":
    # Quick manual check against AlexNet. Close other apps first to reduce OS-level noise.
    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print("[INFO] Profiling compute time...")
    compute_stats = profile_chunk("models/alexnet.onnx", sample)
    _print_stats("compute", compute_stats)

    print("[INFO] Measuring PCIe overhead...")
    pcie_stats = measure_pcie_overhead("models/alexnet.onnx", sample)
    _print_stats("upload  ", pcie_stats["upload"])
    _print_stats("download", pcie_stats["download"])