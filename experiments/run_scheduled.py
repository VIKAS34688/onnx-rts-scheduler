"""
Scheduled run: same task set as run_baseline.py, but routed through
src/scheduler.py's dispatcher (Rules 1-5) with chunks produced by
src/splitter.py and validated against src/schedulability.py.

TODO (Person C):
- Load the split AlexNet chunk .onnx files
- Wrap each task's chunk sequence into ChunkRequest objects
- Submit to the Scheduler and run for N hyperperiods
- Record: completion times, deadline misses, and compare against
  run_baseline.py's results for the same task set
"""

if __name__ == "__main__":
    raise NotImplementedError
