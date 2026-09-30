"""
Naive baseline: concurrent AlexNet inference with no priority-aware
scheduling, no chunk splitting — the "before" picture, matching the
paper's own Baseline comparison in Section VIII.

TODO (Person C):
- Load the un-split AlexNet.onnx model
- Spin up one InferenceSession per task in the generated task set
- Run all tasks concurrently for N hyperperiods with no coordination
- Record: completion times, deadline misses
"""

if __name__ == "__main__":
    raise NotImplementedError
