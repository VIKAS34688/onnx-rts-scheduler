"""
Export the pretrained YOLOv8-nano detector to ONNX. The export has no non-maximum
suppression inside the graph, so NMS can be timed on the CPU as a separate step.

Setup (once):   pip install ultralytics
Usage:          python scripts/export_yolo.py
"""

import os
import shutil

import onnx
from ultralytics import YOLO

os.makedirs("models", exist_ok=True)
model = YOLO("yolov8n.pt")            # downloads the pretrained weights (about 6 MB) on first use
exported = model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=False)
shutil.move(exported, "models/yolov8n.onnx")

m = onnx.load("models/yolov8n.onnx")
shp = lambda vi: [d.dim_value for d in vi.type.tensor_type.shape.dim]
print(f"[DONE] models/yolov8n.onnx  input {m.graph.input[0].name} {shp(m.graph.input[0])}")
for o in m.graph.output:
    print(f"       output {o.name} {shp(o)}")
print(f"       {len(m.graph.node)} nodes")
