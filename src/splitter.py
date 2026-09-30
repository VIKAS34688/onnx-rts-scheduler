"""
Graph splitting for ONNX models.

Mirrors the reference paper's Section VII splitting mechanics, but uses
ONNX's own graph-extraction API instead of torch.fx / Caffe .prototxt
parsing:

    onnx.utils.extract_model(input_path, output_path, input_names, output_names)

Candidate split points, per the paper, are locations in the graph where
"no other node can run in parallel" — i.e. strictly sequential regions.
For AlexNet (Phase 1's model) this holds for the entire network, since it
has no skip connections or branching.

TODO (Person B):
- Load the exported AlexNet .onnx model
- Walk the graph (onnx.load + inspect model.graph.node) to enumerate
  candidate split points (boundaries between sequential nodes)
- Given a chosen set of active split points, call extract_model() once
  per chunk boundary pair to produce the chunk sub-models
- Save each chunk as its own .onnx file under models/chunks/
"""

import onnx


def list_candidate_split_points(onnx_model_path: str):
    """
    Return an ordered list of tensor names that are valid split boundaries.

    TODO: implement the graph walk described in the module docstring.
    For a fully sequential model, every node's output tensor is a
    candidate boundary.
    """
    raise NotImplementedError


def extract_chunk(onnx_model_path: str, output_path: str,
                   input_tensor_names: list, output_tensor_names: list):
    """
    Thin wrapper around onnx.utils.extract_model for one chunk.
    """
    onnx.utils.extract_model(
        onnx_model_path, output_path,
        input_names=input_tensor_names,
        output_names=output_tensor_names,
    )


def build_chunks(onnx_model_path: str, active_split_points: list, out_dir: str):
    """
    Given a set of activated split points, produce the sequence of chunk
    .onnx files: first chunk = start -> first split point, middle chunks
    = between consecutive split points, last chunk = last split point -> end.

    TODO (Person B): implement using extract_chunk() above.
    """
    raise NotImplementedError
