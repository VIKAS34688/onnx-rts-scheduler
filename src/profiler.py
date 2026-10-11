"""
Per-chunk timing profiling.

The reference paper profiles engine execution times from the CPU side
using high-resolution wall-clock timers, isolating each chunk (running
it alone on the GPU) to get clean WCET estimates uncontaminated by
resource contention. We follow the same wall-clock approach here.

On our discrete GPU (unlike the paper's unified-memory Jetson boards),
input/output data has to physically cross the PCIe bus between host
(CPU) RAM and device (GPU) VRAM. We use ONNX Runtime's IOBinding to
keep data resident on the GPU across the actual compute call, which
lets us measure:

  - compute time   -- the GPU-only number, comparable to the paper's
  - upload time    -- host -> device copy (does not exist in the
                       paper's unified-memory setup)
  - download time  -- device -> host copy (same)

NOTE on outliers: our GTX 1650 is also rendering the Windows desktop
while we profile -- the paper's embedded Jetson boards carry no such
background load. We report percentiles (p50/p95/p99) alongside min/
mean/max so a rare OS-level stall isn't mistaken for the true worst case.

NOTE on chunk inputs: only chunk 0 takes the image. Later chunks take an
intermediate tensor with a fixed shape, so for any chunk whose declared
input shape differs from the sample we build a random tensor of the right
shape. Convolution/Gemm timing does not depend on the values.

NOTE on warmup: after an idle gap (for example while chunk files are being
written) the GPU sits at low clocks, and a handful of ~0.3 ms runs finish
before the clocks ramp up. Warmup is therefore time-based as well as
count-based.
"""

import torch  # must be imported first -- registers PyTorch's bundled CUDA DLLs for Windows to find
import time
import numpy as np
import onnxruntime as ort

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


def _input_for_session(session, sample_input: np.ndarray) -> np.ndarray:
    """Return sample_input if it fits this chunk, else a random tensor of the chunk's own input shape."""
    declared = session.get_inputs()[0].shape
    shape = [d if isinstance(d, int) else 1 for d in declared]
    if list(sample_input.shape) != shape:
        return np.random.randn(*shape).astype(np.float32)
    return sample_input


def profile_chunk(chunk_onnx_path: str, sample_input: np.ndarray,
                   n_runs: int = 300, warmup: int = 10) -> dict:
    """
    Run a single chunk n_runs times in isolation and return compute-only
    timing stats (min/mean/p50/p95/p99/max, in milliseconds).

    Input and output are kept GPU-resident via IOBinding for the timed
    region, so this number reflects compute time alone -- NOT including
    the PCIe upload/download cost. Use measure_pcie_overhead() for that.
    """
    session = ort.InferenceSession(
        chunk_onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sample_input = _input_for_session(session, sample_input)

    # Upload once, outside the timed loop -- we're isolating compute here.
    gpu_input = ort.OrtValue.ortvalue_from_numpy(sample_input, "cuda", 0)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(input_name, gpu_input)
    io_binding.bind_output(output_name, "cuda")  # keep output on GPU too

    # Warm up by TIME, not run count: after an idle gap the GPU sits at low
    # clocks, and ~10 tiny runs are over before clocks ramp up.
    warm_until = time.perf_counter() + WARMUP_SECONDS
    runs = 0
    while runs < warmup or time.perf_counter() < warm_until:
        session.run_with_iobinding(io_binding)
        runs += 1

    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run_with_iobinding(io_binding)  # compute only, no copy
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return _percentile_stats(times_ms)


def measure_pcie_overhead(chunk_onnx_path: str, sample_input: np.ndarray,
                           n_runs: int = 300, warmup: int = 10) -> dict:
    """
    Measure the upload (host->device) and download (device->host) costs
    SEPARATELY from compute time -- the overhead the paper's Jetson setup
    does not have, that we committed to reporting honestly.
    """
    session = ort.InferenceSession(
        chunk_onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sample_input = _input_for_session(session, sample_input)

    for _ in range(warmup):
        gpu_input = ort.OrtValue.ortvalue_from_numpy(sample_input, "cuda", 0)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_name, gpu_input)
        io_binding.bind_output(output_name, "cuda")
        session.run_with_iobinding(io_binding)
        _ = io_binding.get_outputs()[0].numpy()

    upload_times_ms, download_times_ms = [], []

    for _ in range(n_runs):
        # --- upload: host -> device ---
        start = time.perf_counter()
        gpu_input = ort.OrtValue.ortvalue_from_numpy(sample_input, "cuda", 0)
        end = time.perf_counter()
        upload_times_ms.append((end - start) * 1000.0)

        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_name, gpu_input)
        io_binding.bind_output(output_name, "cuda")
        session.run_with_iobinding(io_binding)  # compute, not timed here

        # --- download: device -> host ---
        gpu_output = io_binding.get_outputs()[0]
        start = time.perf_counter()
        _ = gpu_output.numpy()  # triggers the actual device->host copy
        end = time.perf_counter()
        download_times_ms.append((end - start) * 1000.0)

    return {
        "upload": _percentile_stats(upload_times_ms),
        "download": _percentile_stats(download_times_ms),
    }


if __name__ == "__main__":
    # Quick manual check against AlexNet. Close other apps first to
    # reduce OS-level noise in the results.
    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print("[INFO] Profiling compute time...")
    compute_stats = profile_chunk("models/alexnet.onnx", sample)
    _print_stats("compute", compute_stats)

    print("[INFO] Measuring PCIe overhead...")
    pcie_stats = measure_pcie_overhead("models/alexnet.onnx", sample)
    _print_stats("upload  ", pcie_stats["upload"])
    _print_stats("download", pcie_stats["download"])