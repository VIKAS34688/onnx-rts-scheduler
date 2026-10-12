"""
Chooses where ONNX Runtime tensors live. Default: the GPU.
Set the environment variable RTS_DEVICE=cpu to debug on a machine
without CUDA (for example a teammate's laptop).
"""

import os

DEVICE = os.environ.get("RTS_DEVICE", "cuda")

if DEVICE == "cuda":
    import torch  # noqa: F401  -- must come first on Windows: registers PyTorch's bundled CUDA DLLs

PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda"
             else ["CPUExecutionProvider"])