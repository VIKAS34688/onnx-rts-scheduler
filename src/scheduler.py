"""
Runtime dispatcher enforcing the reference paper's Rules 1-5 (Section V),
adapted to ONNX Runtime instead of TensorRT/CUDA streams.

Rule 1 — At most one chunk engine executes (or is pending) on the GPU
         at any time.
Rule 2 — Pending requests are served strictly by task priority.
Rule 3 — A running chunk is never preempted mid-execution.
Rule 4 — Chunks of one DNN run sequentially, output feeding into input.
Rule 5 — After a chunk finishes, the task keeps the GPU only if it has
         more chunks left AND no higher-priority task is waiting.

IMPORTANT: do not assume ONNX Runtime's session.run() being synchronous
gives you Rule 1 for free. Verify it empirically (CUDA event timers,
checking for overlapping GPU activity) before relying on it — see
profiler.py.

TODO (Person A):
- Implement a thread-safe priority queue of pending chunk requests
- Implement the dispatcher loop: pop highest-priority request, run its
  chunk (session.run()), on completion check Rule 5 before deciding
  whether to continue with the next chunk of the same task or yield
  the GPU to a newly-arrived higher-priority request
"""

import heapq
import threading


class ChunkRequest:
    def __init__(self, task_priority: int, task_id: str, chunk_index: int,
                 session, input_tensor):
        self.task_priority = task_priority
        self.task_id = task_id
        self.chunk_index = chunk_index
        self.session = session
        self.input_tensor = input_tensor

    def __lt__(self, other):
        # Lower number = higher priority, matching Deadline-Monotonic
        # convention used in the paper.
        return self.task_priority < other.task_priority


class Scheduler:
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()

    def submit(self, request: ChunkRequest):
        """Enqueue a chunk request. TODO: wake the dispatcher if idle."""
        with self._lock:
            heapq.heappush(self._queue, request)

    def run_dispatcher(self):
        """
        Main dispatch loop enforcing Rules 1-5.

        TODO (Person A): implement the loop described in the module
        docstring. Empirically confirm Rule 1 holds using the timing
        utilities in profiler.py before trusting this in the experiment.
        """
        raise NotImplementedError
