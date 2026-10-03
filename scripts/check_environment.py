"""
Day 1-3 setup check.

Run this first, before anything else, to confirm:
  1. PyTorch sees the GTX 1650 (CUDA available)
  2. ONNX Runtime has the CUDA execution provider available
  3. The GPU's native CUDA stream priority range (answers the open
     question flagged during planning -- stop guessing, get the real
     number)

Usage:
    python scripts/check_environment.py
"""

import sys


def check_pytorch_cuda():
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed. Run: pip install -r requirements.txt")
        return False

    available = torch.cuda.is_available()
    print(f"[INFO] torch.cuda.is_available(): {available}")
    if available:
        print(f"[INFO] GPU name: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version (PyTorch build): {torch.version.cuda}")
    else:
        print("[FAIL] PyTorch cannot see a CUDA GPU. Check your NVIDIA driver "
              "and that you installed the CUDA-enabled PyTorch build.")
    return available


def check_onnxruntime_cuda():
    try:
        import onnxruntime as ort
    except ImportError:
        print("[FAIL] onnxruntime is not installed. "
              "Run: pip install onnxruntime-gpu")
        return False

    providers = ort.get_available_providers()
    print(f"[INFO] ONNX Runtime available providers: {providers}")
    has_cuda = "CUDAExecutionProvider" in providers
    if not has_cuda:
        print("[FAIL] CUDAExecutionProvider not found. You likely have the "
              "CPU-only 'onnxruntime' package installed instead of "
              "'onnxruntime-gpu' -- uninstall one, install the other.")
    return has_cuda


def check_stream_priority_range():
    """
    Queries the GPU's native CUDA stream priority range directly, via
    PyCUDA if available, otherwise falls back to a ctypes call into the
    CUDA runtime so this works with no extra dependency beyond the
    NVIDIA driver itself.
    """
    try:
        import ctypes

        cudart = ctypes.CDLL("cudart64_12.dll") if sys.platform == "win32" \
            else ctypes.CDLL("libcudart.so")

        least = ctypes.c_int()
        greatest = ctypes.c_int()
        ret = cudart.cudaDeviceGetStreamPriorityRange(
            ctypes.byref(least), ctypes.byref(greatest)
        )
        if ret != 0:
            print(f"[FAIL] cudaDeviceGetStreamPriorityRange returned error code {ret}")
            return None

        # Per CUDA semantics: lower numeric value = higher priority.
        num_levels = least.value - greatest.value + 1
        print(f"[INFO] CUDA stream priority range: least={least.value}, "
              f"greatest={greatest.value}  ->  {num_levels} priority level(s) "
              f"available on this GPU")
        return num_levels
    except OSError as e:
        print(f"[FAIL] Could not load the CUDA runtime library: {e}")
        print("        Confirm the CUDA toolkit is installed and on your PATH.")
        return None


if __name__ == "__main__":
    print("=== Day 1-3 environment check ===\n")
    ok_torch = check_pytorch_cuda()
    print()
    ok_ort = check_onnxruntime_cuda()
    print()
    check_stream_priority_range()
    print()

    if ok_torch and ok_ort:
        print("[PASS] Environment looks ready. Proceed to export_alexnet.py.")
    else:
        print("[ACTION NEEDED] Fix the failures above before continuing.")