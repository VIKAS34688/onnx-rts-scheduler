
from pathlib import Path
import netron

# Project root is the parent of the scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the exported AlexNet ONNX model.
MODEL_PATH = PROJECT_ROOT / "models" / "alexnet.onnx"

# Stop if the model does not exist.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"AlexNet ONNX model not found: {MODEL_PATH}\n"
        "Run scripts/export_alexnet.py first."
    )

print(f"Opening ONNX model: {MODEL_PATH}")

# Start the local Netron server and open the graph in a browser.
netron.start(str(MODEL_PATH), browse=True)
