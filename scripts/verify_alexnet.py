import torch  # add this line first — registers PyTorch's bundled CUDA DLLs for Windows to find
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "models/alexnet.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print(f"[INFO] Running on: {session.get_providers()[0]}")

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": dummy_input})

print(f"[INFO] Output shape: {outputs[0].shape}")
print("[PASS] Model loads and runs correctly on ONNX Runtime.")