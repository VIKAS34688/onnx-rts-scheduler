"""
Per-chunk timing profiling.

The reference paper profiles engine execution times from the CPU side
using high-resolution wall-clock timers, isolating each chunk (running
it alone on the GPU) to get clean WCET estimates uncontaminated by
resource contention.

On our discrete GPU (unlike the paper's unified-memory Jetson boards),
we additionally need to separately measure the PCIe copy overhead for
the very first input upload and final output download — this cost does
not exist in the paper's setup and must be disclosed, not hidden.

TODO (Person A):
- Load a chunk as an onnxruntime.InferenceSession (CUDA execution provider)
- Run it N times in isolation, recording wall-clock time per run
- Report min / mean / max as the empirical WCET estimate
- Separately, wrap the input-upload and output-download memcopy calls
  with CUDA event timers (or wall-clock, if using IOBinding) to get a
  standalone PCIe overhead number
"""

import time
import onnxruntime as ort


def profile_chunk(chunk_onnx_path: str, sample_input, n_runs: int = 100):
    """
    Run a single chunk n_runs times in isolation and return timing stats.

    TODO: implement using onnxruntime.InferenceSession with
    providers=["CUDAExecutionProvider"], and IOBinding to avoid
    unnecessary host<->device copies between repeated runs.
    """
    raise NotImplementedError


def measure_pcie_overhead(chunk_onnx_path: str, sample_input):
    """
    Measure the cost of the input upload (host -> device) and output
    download (device -> host) separately from compute time.

    TODO: use CUDA events or IOBinding's copy_outputs_to_cpu() timing
    to isolate this from the profile_chunk() compute-only number.
    """
    raise NotImplementedError
