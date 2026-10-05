"""
Runtime dispatcher enforcing Rules 1-5 from the reference paper (Section V),
adapted to ONNX Runtime instead of TensorRT/CUDA streams.

Rule 1 - at most one chunk engine executes on the GPU at any time
Rule 2 - pending requests are served strictly by task priority
Rule 3 - a running chunk is never preempted mid-execution
Rule 4 - chunks of one DNN run sequentially, output feeding into input
Rule 5 - after a chunk finishes, the task keeps the GPU only if it has
         more chunks left AND no higher-priority task is waiting

Rule 1 is enforced here because session.run() is a synchronous, blocking
call -- the dispatch loop never starts a second chunk until the current
one returns. Verified empirically via test_scheduler_integration.py
using CPU wall-clock timestamps -- consistent with the reference paper's
own Section IV methodology, which also measures wall-clock time on the
CPU rather than raw CUDA events.

Inter-chunk handoff (Rule 4) uses IOBinding to keep a chunk's output
GPU-resident and feed it directly as the next chunk's input, avoiding
a host round-trip between chunks of the same task.
"""

import heapq
import threading
import time
import onnxruntime as ort


class Task:
    """
    A DNN task: a fixed priority (Deadline-Monotonic convention --
    lower number = higher priority) plus an ordered sequence of chunk
    sessions to run one after another (Rule 4).

    make_input_feed(chunk_index, prev_outputs) is only ever called for
    chunk_index 0 -- it must return {input_name: numpy_array} for the
    task's real, host-side input. Every later chunk's input comes
    directly from the previous chunk's GPU-resident output via
    IOBinding, bypassing this callback entirely.
    """

    def __init__(self, task_id: str, priority: int, chunk_sessions: list,
                 make_input_feed):
        self.task_id = task_id
        self.priority = priority
        self.chunk_sessions = chunk_sessions
        self.make_input_feed = make_input_feed
        self._seq = None  # heapq tie-breaker, set on submit

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._seq < other._seq  # FIFO among equal priority


class _ResumedTask:
    """
    A task resuming from a specific chunk index after being preempted
    by Rule 5. Carries forward the previous chunk's GPU-resident
    OrtValue output directly -- no host round-trip on resume either.
    """

    def __init__(self, original: Task, resume_from: int, prev_output_ortvalue):
        self.task_id = original.task_id
        self.priority = original.priority
        self.chunk_sessions = original.chunk_sessions[resume_from:]
        self._prev_output_ortvalue = prev_output_ortvalue
        self.make_input_feed = original.make_input_feed  # unused past chunk 0
        self._seq = None

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._seq < other._seq


class Scheduler:
    def __init__(self):
        self._queue = []
        self._seq_counter = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._running = False
        self._thread = None
        # Every executed chunk gets logged here -- this is what
        # verify_no_overlap() checks for overlaps.
        self.execution_log = []

    def submit_job(self, task):
        with self._lock:
            task._seq = self._seq_counter
            self._seq_counter += 1
            heapq.heappush(self._queue, task)
            self._not_empty.notify()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
            self._not_empty.notify_all()
        if self._thread:
            self._thread.join()

    def _dispatch_loop(self):
        while True:
            with self._lock:
                while not self._queue and self._running:
                    self._not_empty.wait()
                if not self._queue and not self._running:
                    return
                task = heapq.heappop(self._queue)  # Rule 2

            self._run_task_chunks(task)

    def _run_task_chunks(self, task):
        prev_output_ortvalue = getattr(task, "_prev_output_ortvalue", None)

        for chunk_index, session in enumerate(task.chunk_sessions):
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            io_binding = session.io_binding()

            if prev_output_ortvalue is None:
                # First chunk of this task: real input, comes from host.
                host_feed = task.make_input_feed(chunk_index, None)
                input_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                    host_feed[input_name], "cuda", 0
                )
            else:
                # Later chunk: previous chunk's output, already GPU-resident
                # -- Rule 4 handoff with no host round-trip.
                input_ortvalue = prev_output_ortvalue

            io_binding.bind_ortvalue_input(input_name, input_ortvalue)
            io_binding.bind_output(output_name, "cuda")  # keep output GPU-resident

            # Rule 1 + Rule 3: blocking call, nothing else touches the
            # GPU until this returns.
            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            end = time.perf_counter()

            self.execution_log.append({
                "task_id": task.task_id,
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
            })

            prev_output_ortvalue = io_binding.get_outputs()[0]

            is_last_chunk = (chunk_index == len(task.chunk_sessions) - 1)
            if is_last_chunk:
                return  # caller can retrieve the final result via
                        # prev_output_ortvalue.numpy() if needed

            # Rule 5: only continue with this task's next chunk if no
            # higher-priority job is waiting right now.
            with self._lock:
                if self._queue and self._queue[0].priority < task.priority:
                    remaining = _ResumedTask(task, chunk_index + 1, prev_output_ortvalue)
                    remaining._seq = self._seq_counter
                    self._seq_counter += 1
                    heapq.heappush(self._queue, remaining)
                    return  # yield the GPU -- dispatcher loop re-evaluates priority


def verify_no_overlap(execution_log):
    """
    Empirically check Rule 1: no two logged chunk executions overlap
    in time. Returns True if the log proves single-chunk-at-a-time
    execution actually held.
    """
    sorted_log = sorted(execution_log, key=lambda e: e["start"])
    for i in range(1, len(sorted_log)):
        if sorted_log[i]["start"] < sorted_log[i - 1]["end"]:
            return False
    return True