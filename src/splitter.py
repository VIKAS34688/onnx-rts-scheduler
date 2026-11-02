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

from device import PROVIDERS
from graph_cuts import chunk_interfaces, legal_cut_points, pick_balanced_cuts


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


def pick_multi_cuts(onnx_model_path: str, k: int) -> list:
    """Cut tokens "@<node index>" for k cuts that balance the estimated work (see graph_cuts.pick_balanced_cuts)."""
    cuts, _, _ = pick_balanced_cuts(onnx.load(onnx_model_path), k)
    return [f"@{i}" for i in cuts]


def build_chunks(onnx_model_path: str, active_split_points: list, out_dir: str) -> list:
    """
    Given activated cuts (in graph order), produce the chunk .onnx files.

    Two kinds of cut:
      tensor names ("/features/..."): exactly one tensor crosses the cut (legal cuts, as in the paper)
      "@<n>": a cut after node n that MAY carry several tensors. The chunks are node ranges, a chunk reads
              every earlier tensor it needs and writes every tensor a later chunk still needs; the scheduler
              keeps these tensors on the GPU between chunks.

    Zero cuts => a single chunk identical to the unsplit model.
    """
    model = onnx.load(onnx_model_path)
    os.makedirs(out_dir, exist_ok=True)

    if active_split_points and str(active_split_points[0]).startswith("@"):
        cut_nodes = sorted(int(str(t)[1:]) for t in active_split_points)
        chunk_paths = []
        for i, (a, b, inputs, outputs) in enumerate(chunk_interfaces(model, cut_nodes)):
            chunk_path = os.path.join(out_dir, f"chunk_{i}.onnx")
            extract_chunk(onnx_model_path, chunk_path, inputs, outputs)
            chunk_paths.append(chunk_path)
        return chunk_paths

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


def run_chunk_chain(chunk_paths: list, sample_input: np.ndarray):
    """Run the chunks one after another (each reads the tensors it needs from what earlier chunks made)."""
    store = {}
    final = None
    for k, path in enumerate(chunk_paths):
        session = ort.InferenceSession(path, providers=PROVIDERS)
        if k == 0:
            feed = {session.get_inputs()[0].name: sample_input}
        else:
            feed = {i.name: store[i.name] for i in session.get_inputs()}
        outs = session.run(None, feed)
        for meta, value in zip(session.get_outputs(), outs):
            store[meta.name] = value
        final = outs[0]
    return final


def verify_chunks_match_original(original_path: str, chunk_paths: list,
                                   sample_input: np.ndarray, atol: float = 1e-4, rtol: float = 1e-3) -> bool:
    """
    Correctness check: run the chunks in order and compare the final result against
    running the original, unsplit model directly.
    """
    orig_session = ort.InferenceSession(original_path, providers=PROVIDERS)
    orig_output = orig_session.run(None, {orig_session.get_inputs()[0].name: sample_input})[0]
    chained = run_chunk_chain(chunk_paths, sample_input)
    matches = np.allclose(orig_output, chained, atol=atol, rtol=rtol)
    max_diff = float(np.max(np.abs(orig_output - chained)))
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