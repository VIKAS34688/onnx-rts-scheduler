"""
Graph splitting for ONNX models.

Mirrors the reference paper's Section VII splitting mechanics, using
ONNX's own graph-extraction API (onnx.utils.extract_model) instead of
torch.fx / Caffe .prototxt parsing.

Candidate split points are the LEGAL cuts of the graph (see graph_cuts.py): places
where exactly one tensor crosses. For AlexNet, a straight chain, that is every node
output except the last, so the list is the same 19 points as before.
"""

import os
import onnx
import onnxruntime as ort
import numpy as np

from graph_cuts import legal_cut_points


def _model_input_name(model) -> str:
    names = [i.name for i in model.graph.input if i.name not in {w.name for w in model.graph.initializer}]
    if len(names) != 1:
        raise ValueError(f"this pipeline handles models with exactly one input, found {len(names)}: {names}")
    return names[0]


def _model_output_name(model) -> str:
    if len(model.graph.output) != 1:
        raise ValueError("this pipeline handles models with exactly one output, found "
                         f"{len(model.graph.output)}: {[o.name for o in model.graph.output]}")
    return model.graph.output[0].name


def list_candidate_split_points(onnx_model_path: str) -> list:
    """Names of the tensors where the graph can legally be cut, in graph order."""
    model = onnx.load(onnx_model_path)
    return [tensor for _, tensor in legal_cut_points(model)]


def extract_chunk(onnx_model_path: str, output_path: str,
                   input_tensor_names: list, output_tensor_names: list):
    """Thin wrapper around onnx.utils.extract_model for one chunk."""
    onnx.utils.extract_model(
        onnx_model_path, output_path,
        input_names=input_tensor_names,
        output_names=output_tensor_names,
    )


def build_chunks(onnx_model_path: str, active_split_points: list, out_dir: str) -> list:
    """
    Given a set of activated split points (in graph order), produce the
    sequence of chunk .onnx files:
      first chunk  = model input -> first active split point
      middle chunk = between consecutive active split points
      last chunk   = last active split point -> model output

    Zero active split points => a single chunk identical to the
    unsplit model (used as the baseline "no split needed" case).
    """
    model = onnx.load(onnx_model_path)
    model_input = _model_input_name(model)
    model_output = _model_output_name(model)

    os.makedirs(out_dir, exist_ok=True)

    boundaries = [model_input] + list(active_split_points) + [model_output]
    chunk_paths = []
    for i in range(len(boundaries) - 1):
        chunk_path = os.path.join(out_dir, f"chunk_{i}.onnx")
        extract_chunk(onnx_model_path, chunk_path,
                      [boundaries[i]], [boundaries[i + 1]])
        chunk_paths.append(chunk_path)
    return chunk_paths


def verify_chunks_match_original(original_path: str, chunk_paths: list,
                                   sample_input: np.ndarray, atol: float = 1e-4) -> bool:
    """
    Correctness check: run the chunks sequentially (chunk 1's output
    feeding chunk 2's input, etc.) and compare the final result against
    running the original, unsplit model directly.
    """
    orig_session = ort.InferenceSession(
        original_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    orig_input_name = orig_session.get_inputs()[0].name
    orig_output = orig_session.run(None, {orig_input_name: sample_input})[0]

    current = sample_input
    for chunk_path in chunk_paths:
        session = ort.InferenceSession(
            chunk_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        current = session.run(None, {input_name: current})[0]

    matches = np.allclose(orig_output, current, atol=atol)
    max_diff = float(np.max(np.abs(orig_output - current)))
    print(f"[INFO] Max difference between original and chunked output: {max_diff:.8f}")
    return matches


if __name__ == "__main__":
    import torch  # import first -- registers PyTorch's bundled CUDA DLLs

    model_path = "models/alexnet.onnx"

    print("[INFO] Finding candidate split points...")
    candidates = list_candidate_split_points(model_path)
    print(f"[INFO] Found {len(candidates)} candidate split points")
    for i, name in enumerate(candidates):
        print(f"  [{i}] {name}")

    # Demo: activate 3 split points roughly spread through the network,
    # producing 4 chunks.
    n = len(candidates)
    demo_points = [candidates[n // 4], candidates[n // 2], candidates[3 * n // 4]]
    print(f"\n[INFO] Building a demo 4-chunk split using: {demo_points}")

    chunk_paths = build_chunks(model_path, demo_points, "models/chunks_demo")
    print(f"[INFO] Built {len(chunk_paths)} chunks: {chunk_paths}")

    print("\n[INFO] Verifying chunked output matches the original model...")
    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)
    matches = verify_chunks_match_original(model_path, chunk_paths, sample)
    print(f"[{'PASS' if matches else 'FAIL'}] Chunked pipeline matches original: {matches}")