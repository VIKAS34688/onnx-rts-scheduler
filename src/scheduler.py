"""
Runtime dispatcher enforcing Rules 1-5 from the reference paper (Section V),
adapted to ONNX Runtime instead of TensorRT/CUDA streams.

Rule 1 - at most one chunk engine executes on the GPU at any time
Rule 2 - pending requests are served strictly by task priority
Rule 3 - a running chunk is never preempted mid-execution
Rule 4 - chunks of one DNN run sequentially, output feeding into input
Rule 5 - after a chunk finishes, the task keeps the GPU only if it has
         more chunks left AND no higher-priority task is waiting

Rule 1 holds because session.run_with_iobinding() is a blocking call and
only the dispatcher thread ever calls it (verified empirically with
verify_no_overlap()).

Hand-over between chunks: every job keeps a "store" of the tensors that its later chunks
still need, on the device, by name. A chunk reads the tensors it needs from the store and
adds the tensors it makes (no host round-trip). With one tensor per cut (the paper's cuts)
the store holds exactly one tensor; with multi-tensor cuts it holds several, including
tensors that skip over a chunk, such as the backbone features a detector's neck reads late.

A job's first chunk takes its input from make_input_feed(); the returned value may be a
numpy array (uploaded here) or an already-uploaded OrtValue (used as is).
on_complete(finish_time) is called once, after the last chunk of a job, with a
time.perf_counter() timestamp.
"""

import heapq
import threading
import time
import onnxruntime as ort

from device import DEVICE


class Task:
    """
    One job: a fixed priority (lower number = higher priority, Deadline-
    Monotonic convention) plus an ordered list of chunk sessions (Rule 4).
    """

    def __init__(self, task_id, priority, chunk_sessions, make_input_feed,
                 on_complete=None):
        self.task_id = task_id
        self.priority = priority
        self.chunk_sessions = chunk_sessions
        self.make_input_feed = make_input_feed
        self.on_complete = on_complete
        self.index_offset = 0
        self._seq = None  # FIFO tie-breaker, set on submit

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._seq < other._seq


class _ResumedTask:
    """
    A job resuming after a Rule-5 preemption. Carries the store of device tensors its remaining
    chunks need, and keeps the original submission order (_seq) so it is not overtaken by later
    jobs of the same priority.
    """

    def __init__(self, original, resume_from, store):
        self.task_id = original.task_id
        self.priority = original.priority
        self.chunk_sessions = original.chunk_sessions[resume_from:]
        self._store = store
        self.make_input_feed = original.make_input_feed
        self.on_complete = original.on_complete
        self.index_offset = original.index_offset + resume_from
        self._seq = original._seq

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
        self.execution_log = []   # one entry per executed chunk
        self.preemptions = 0      # how many times Rule 5 made a job yield

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
        store = dict(getattr(task, "_store", {}))
        sessions = task.chunk_sessions
        feed = None

        for chunk_index, session in enumerate(sessions):
            io_binding = session.io_binding()
            for meta in session.get_inputs():
                if meta.name in store:
                    value = store[meta.name]                     # made by an earlier chunk: already on the device
                else:                                            # the job's real input (first chunk only)
                    if feed is None:
                        feed = task.make_input_feed(chunk_index, None)
                    value = feed[meta.name]
                    if not isinstance(value, ort.OrtValue):
                        value = ort.OrtValue.ortvalue_from_numpy(value, DEVICE, 0)
                io_binding.bind_ortvalue_input(meta.name, value)
            out_names = [m.name for m in session.get_outputs()]
            for name in out_names:
                io_binding.bind_output(name, DEVICE)

            # Rule 1 + Rule 3: blocking call; nothing else touches the GPU until this returns.
            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            end = time.perf_counter()

            self.execution_log.append({
                "task_id": task.task_id,
                "chunk_index": chunk_index + task.index_offset,
                "start": start,
                "end": end,
            })

            for name, value in zip(out_names, io_binding.get_outputs()):
                store[name] = value
            needed_later = {m.name for s in sessions[chunk_index + 1:] for m in s.get_inputs()}
            store = {k: v for k, v in store.items() if k in needed_later}   # free what no later chunk reads

            if chunk_index == len(sessions) - 1:
                if task.on_complete is not None:
                    task.on_complete(end)
                return

            # Rule 5: continue with the next chunk only if nobody with a
            # higher priority is waiting; otherwise yield the device.
            with self._lock:
                if self._queue and self._queue[0].priority < task.priority:
                    self._seq_counter += 1
                    heapq.heappush(self._queue, _ResumedTask(task, chunk_index + 1, store))
                    self.preemptions += 1
                    return


def verify_no_overlap(execution_log):
    """Rule 1 check: no two logged chunk executions overlap in time."""
    s = sorted(execution_log, key=lambda e: e["start"])
    return all(s[i]["start"] >= s[i - 1]["end"] for i in range(1, len(s)))