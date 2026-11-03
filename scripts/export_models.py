"""
Export the paper's other classifiers to ONNX (AlexNet is done by export_alexnet.py)
and check each export against PyTorch.

Usage (project root, venv active; downloads the pretrained weights the first time):
    python scripts/export_models.py                 (vgg19, resnet18, inception_v4)
    python scripts/export_models.py resnet18        (just one)
    python scripts/export_models.py googlenet mobilenet_v2 vgg16 densenet121   (the paper's second set)

InceptionV4 comes from the 'timm' package (pip install timm); torchvision has none.
"""

import os
import sys

import torch  # first: registers PyTorch's bundled CUDA DLLs on Windows
import numpy as np
import onnxruntime as ort
import torchvision.models as tvm


def build(name):
    if name == "vgg19":
        return tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT), 224
    if name == "resnet18":
        return tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT), 224
    if name == "inception_v4":
        import timm
        return timm.create_model("inception_v4", pretrained=True), 299
    # the paper's second model set for Figure 1
    if name == "googlenet":
        return tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT), 224
    if name == "mobilenet_v2":
        return tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT), 224
    if name == "vgg16":
        return tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT), 224
    if name == "densenet121":
        return tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT), 224
    raise SystemExit(f"unknown model: {name}")


def export(name):
    model, size = build(name)
    model.eval()
    path = f"models/{name}.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"],
                      opset_version=17, do_constant_folding=True)

    session = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = float(np.abs(ref - out).max())
    tol = 1e-3 * max(1.0, float(np.abs(ref).max()))
    print(f"[{'PASS' if diff < tol else 'FAIL'}] {name}: {path} ({os.path.getsize(path) / 1e6:.0f} MB), "
          f"max difference vs PyTorch {diff:.2e}, same top-1: {int(ref.argmax()) == int(out.argmax())}, "
          f"provider {session.get_providers()[0]}")


os.makedirs("models", exist_ok=True)
for name in (sys.argv[1:] or ["vgg19", "resnet18", "inception_v4"]):
    try:
        export(name)
    except Exception as e:
        print(f"[SKIP] {name}: {type(e).__name__}: {e}")
