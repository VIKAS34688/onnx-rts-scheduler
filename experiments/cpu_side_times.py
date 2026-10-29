"""
CPU-side time measurements (the paper's second future-work item) and GPU against CPU:

  1. pre-processing  : decode-free image preparation (resize, pad or crop, normalise)
  2. post-processing : YOLO: decode boxes + confidence filter + non-maximum suppression (NMS);
                       classifier: softmax + top-5
  3. GPU time        : the model on the GPU, with data already on the device, and with
                       the host-device copies included (session.run with numpy input)
  4. CPU-only time   : the same model on the CPU (--cpu-only)

NMS time depends on how many boxes survive the confidence filter, so use REAL images with
different numbers of objects (--images FOLDER). Without images a random picture is used and
the post-processing number is not meaningful.

Usage (project root):
    python experiments/cpu_side_times.py --model models/yolov8n.onnx --kind yolo --images my_images --cpu-only
    python experiments/cpu_side_times.py --model models/alexnet.onnx --kind classifier --images my_images --cpu-only
Results are also saved to docs/phase2/cpu_side_<model>.json
"""

import _common  # noqa: F401  -- must be first: sets up the import path
import argparse
import glob
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from device import DEVICE, PROVIDERS

try:                                   # the usual fast NMS, if available
    import torch
    import torchvision.ops as tv_ops
except Exception:
    tv_ops = None


def stats(times_ms):
    a = np.asarray(times_ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "n": int(a.size)}


def timed(fn, repeats):
    out, times = None, []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def preprocess_yolo(img, size=640):
    """Letterbox: scale to fit, pad with grey (114), then float32 CHW in [0, 1]."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(img.convert("RGB").resize((nw, nh), Image.BILINEAR))
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_cls(img, size=224):
    """Resize shorter side to 256, centre-crop 224, normalise with the ImageNet statistics."""
    img = img.convert("RGB")
    w, h = img.size
    r = 256 / min(w, h)
    img = img.resize((int(round(w * r)), int(round(h * r))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    a = np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def nms_numpy(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thr]
    return np.asarray(keep)


def postprocess_yolo(raw, conf=0.25, iou=0.45):
    """raw: [1, 4 + classes, N] as exported by ultralytics. Returns (boxes before NMS, boxes after NMS)."""
    p = raw[0].T                                     # (N, 4 + classes)
    scores_all = p[:, 4:]
    cls, score = scores_all.argmax(1), scores_all.max(1)
    keep = score > conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if p.shape[0] == 0:
        return 0, 0
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) + cls[:, None] * 4096.0   # class-aware NMS
    if tv_ops is not None:
        kept = tv_ops.nms(torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)),torch.from_numpy(np.ascontiguousarray(score, dtype=np.float32)),iou,).numpy()
    else:
        kept = nms_numpy(boxes, score, iou)
    return int(p.shape[0]), int(len(kept))


def postprocess_cls(raw):
    e = np.exp(raw[0] - raw[0].max())
    prob = e / e.sum()
    return np.argsort(prob)[::-1][:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["yolo", "classifier"], required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    files = []
    if args.images:
        for ext in ("jpg", "jpeg", "png", "bmp"):
            files += glob.glob(os.path.join(args.images, f"*.{ext}"))
    imgs = [Image.open(f) for f in sorted(files)[:args.max_images]]
    if not imgs:
        print("[WARN] no images found: using ONE random picture. Pre-processing is still meaningful, "
              "post-processing (NMS) is NOT.")
        imgs = [Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))]
    print(f"[INFO] {len(imgs)} image(s), NMS backend: {'torchvision' if tv_ops is not None else 'numpy'}")
    pre = (lambda im: preprocess_yolo(im, args.size)) if args.kind == "yolo" else preprocess_cls

    session = ort.InferenceSession(args.model, providers=PROVIDERS)
    in_name = session.get_inputs()[0].name
    pre_t, post_t, gpu_copy_t, boxes = [], [], [], []
    x0 = None
    for im in imgs:
        x, t = timed(lambda: pre(im), args.repeats)
        pre_t += t
        x0 = x if x0 is None else x0
        raw = session.run(None, {in_name: x})[0]
        if args.kind == "yolo":
            res, t = timed(lambda: postprocess_yolo(raw), args.repeats)
            boxes.append(res)
        else:
            _, t = timed(lambda: postprocess_cls(raw), args.repeats)
        post_t += t
        _, t = timed(lambda: session.run(None, {in_name: x}), max(5, args.repeats // 4))
        gpu_copy_t += t

    out = {"model": os.path.basename(args.model), "kind": args.kind, "images": len(imgs),
           "preprocess": stats(pre_t), "postprocess": stats(post_t), "session_run_with_copies": stats(gpu_copy_t)}
    if boxes:
        out["boxes_before_nms_mean"] = float(np.mean([b[0] for b in boxes]))
        out["boxes_after_nms_mean"] = float(np.mean([b[1] for b in boxes]))
    if DEVICE == "cuda":
        import profiler
        out["gpu_resident"] = {k: v for k, v in profiler.profile_chunk(args.model, x0, n_runs=200).items()}
    if args.cpu_only:
        cpu = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        for _ in range(3):
            cpu.run(None, {in_name: x0})
        _, t = timed(lambda: cpu.run(None, {in_name: x0}), 50)
        out["cpu_only_inference"] = stats(t)

    print(f"\n{'step':<34} {'p50':>9} {'p95':>9} {'p99':>9}   (ms)")
    for label, key in (("CPU pre-processing", "preprocess"), ("CPU post-processing", "postprocess"),
                       ("model run incl. host-device copies", "session_run_with_copies"),
                       ("model on GPU, data already there", "gpu_resident"),
                       ("model on CPU only", "cpu_only_inference")):
        if key in out:
            s = out[key]
            print(f"{label:<34} {s['p50_ms']:>9.3f} {s['p95_ms']:>9.3f} {s['p99_ms']:>9.3f}")
    if boxes:
        print(f"\nboxes above the confidence threshold: {out['boxes_before_nms_mean']:.0f} on average, "
              f"{out['boxes_after_nms_mean']:.0f} after NMS")
    p50 = out["preprocess"]["p50_ms"] + out["session_run_with_copies"]["p50_ms"] + out["postprocess"]["p50_ms"]
    p99 = out["preprocess"]["p99_ms"] + out["session_run_with_copies"]["p99_ms"] + out["postprocess"]["p99_ms"]
    print(f"\nend to end (pre + model with copies + post): p50 {p50:.2f} ms, sum of p99s {p99:.2f} ms")
    if "cpu_only_inference" in out and "gpu_resident" in out:
        print(f"GPU speed-up over CPU-only (medians): {out['cpu_only_inference']['p50_ms'] / out['gpu_resident']['p50_ms']:.1f}x")
    os.makedirs("docs/phase2", exist_ok=True)
    json.dump(out, open(f"docs/phase2/cpu_side_{out['model'].replace('.onnx', '')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
