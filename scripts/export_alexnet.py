"""
Export a pretrained AlexNet to ONNX -- the Phase 1 model.

AlexNet is chosen deliberately over ResNet18/VGG19 for this first pass:
it is a single, fully sequential chain of layers with zero skip
connections or branching, which makes graph-splitting with
onnx.utils.extract_model as reliable as possible while the rest of the
pipeline (scheduler, profiler, schedulability loop) is still being
proven out.

Usage:
    python scripts/export_alexnet.py
"""

import os
import torch
import torchvision.models as models


def export_alexnet(output_path: str = "models/alexnet.onnx"):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()

    # AlexNet's native input size: 1 image, 3 color channels, 224x224 pixels.
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[DONE] Exported AlexNet to {output_path}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    export_alexnet()