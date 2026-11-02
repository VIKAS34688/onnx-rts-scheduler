"""
Overnight runner: runs all the remaining GPU jobs one after another, unattended.

Why a script and not an agent: the jobs are fixed commands. This runner does exactly those, logs each
one, and protects the night from the hang you saw at U=0.9 set 5: if a job prints nothing for
--idle-min minutes (a set normally takes about 20 s) it is killed, and the runner moves on to the next job.
Every sweep writes its own CSV, so one bad job costs only that job.

Before you start (charger in, nothing else on the GPU):
  - Power plan: powercfg /change standby-timeout-ac 0     and     powercfg /change monitor-timeout-ac 0
  - Pause OneDrive sync (tray icon), set Best performance, close every other program.
  - Do NOT run anything else on the GPU while this runs, it would ruin the timings.

Usage (project root, venv active):
    python experiments\\overnight.py --dry-run          (show the plan and the time estimate)
    python experiments\\overnight.py                    (run everything)
    python experiments\\overnight.py --only u07 u09     (run only jobs whose name contains one of these)

Logs: docs/phase2/overnight_logs/<job>.log and summary.txt (read summary.txt first in the morning).
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import threading
import time

PY = sys.executable
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
LOG_DIR = os.path.join("docs", "phase2", "overnight_logs")
RES = os.path.join("experiments", "results")
CACHE = os.path.join("models", "profile_cache.json")


def sweep(name, util, method, sets, cands, minutes, idle_min):
    out = os.path.join(RES, f"{name}.csv")
    cmd = [PY, "experiments/run_experiment.py", "--n-tasks", "3", "--snap", "--overhead-ms", "0.35",
           "--utils", str(util), "--sets", str(sets), "--method", method,
           "--max-candidates", str(cands), "--out", out]
    return {"name": name, "cmd": cmd, "idle_min": idle_min, "est_min": minutes, "csv": out}


def build_jobs():
    jobs = [{"name": "cpu_side_times",
             "cmd": [PY, "experiments/cpu_side_times.py", "--model", "models/yolov8n.onnx", "--kind", "yolo",
                     "--images", "my_images", "--cpu-only"],
             "idle_min": 10, "est_min": 3, "csv": None}]
    # redo U=0.7 (the first run had more proposed than baseline misses) and finish U=0.9 (it hung at set 5)
    jobs.append(sweep("alexnet_n3_u07_rerun", 0.7, "heuristic", 50, 7, 20, 10))
    jobs.append(sweep("alexnet_n3_u09", 0.9, "heuristic", 50, 7, 20, 10))
    # heuristic against optimal on the SAME task sets and the same 5 candidate cuts (paper Fig. 5)
    for u in (0.6, 0.7, 0.8, 0.9):
        tag = f"u{int(round(u * 10)):02d}"
        jobs.append(sweep(f"alexnet_n3_h5_{tag}", u, "heuristic", 20, 5, 10, 15))
        jobs.append(sweep(f"alexnet_n3_o5_{tag}", u, "optimal", 20, 5, 30, 40))
    # detector Option B again with the fixed chunk-file names (the old names could reuse another run's chunks)
    yolo_out = os.path.join(RES, "yolo_n3_B_v2.csv")
    jobs.append({"name": "yolo_n3_B_v2", "idle_min": 15, "est_min": 16, "csv": yolo_out,
                 "cmd": [PY, "experiments/run_experiment.py", "--model", "models/yolov8n.onnx", "--multi",
                         "--max-candidates", "7", "--n-tasks", "3", "--utils", "0.6", "0.7", "0.8", "0.9",
                         "--sets", "10", "--snap", "--overhead-ms", "0.35", "--out", yolo_out]})
    # the two new scripts from the agent: first GPU runs, short
    jobs.append({"name": "baseline_sp_demo", "idle_min": 10, "est_min": 2, "csv": None,
                 "cmd": [PY, "experiments/run_baseline_sp.py"]})
    mixed_out = os.path.join(RES, "mixed_quick_test.csv")
    jobs.append({"name": "mixed_quick", "idle_min": 15, "est_min": 5, "csv": mixed_out,
                 "cmd": [PY, "experiments/run_experiment_mixed.py", "--models", "models/alexnet.onnx",
                         "models/resnet18.onnx", "--sets", "2", "--duration", "4.0", "--snap", "--out", mixed_out]})
    return jobs


def kill_tree(p):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        p.kill()


def dump_stack(p, log_path):
    """Before killing a hung job, save where it is waiting (needs: pip install py-spy)."""
    try:
        r = subprocess.run(["py-spy", "dump", "--pid", str(p.pid)], capture_output=True, text=True, timeout=60)
        text = r.stdout + r.stderr
    except Exception as e:
        text = f"py-spy not available or failed: {e}\n"
    with open(log_path + ".stack.txt", "w", encoding="utf-8") as f:
        f.write(text)


def run_job(name, cmd, idle_s, total_s, log_path):
    """Run one command; kill it if it prints nothing for idle_s seconds or runs longer than total_s."""
    start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1, env=ENV)
    last = [time.time()]

    def reader():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in p.stdout:
                last[0] = time.time()
                f.write(line)
                f.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    status = None
    while p.poll() is None:
        time.sleep(5)
        now = time.time()
        if now - last[0] > idle_s:
            status = f"KILLED: no output for {idle_s / 60:.0f} min (hang)"
            dump_stack(p, log_path)
            kill_tree(p)
            break
        if now - start > total_s:
            status = f"KILLED: longer than {total_s / 3600:.1f} h"
            kill_tree(p)
            break
    p.wait()
    t.join(timeout=10)
    if status is None:
        status = "OK" if p.returncode == 0 else f"FAILED (exit code {p.returncode}), see the log"
    return status, time.time() - start


def count_rows(path):
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--total-hours", type=float, default=4.0, help="longest any single job may run")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.only:
        jobs = [j for j in jobs if any(k in j["name"] for k in args.only)]
    print(f"{len(jobs)} jobs, rough total {sum(j['est_min'] for j in jobs) / 60:.1f} h (a guess, the optimal sweeps vary a lot)")
    for j in jobs:
        print(f"  {j['name']:<24} ~{j['est_min']:>3} min   idle limit {j['idle_min']} min")
    if args.dry_run:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    summary = os.path.join(LOG_DIR, "summary.txt")
    started = datetime.datetime.now()
    results = []

    def write_summary(done=False):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(f"started {started:%Y-%m-%d %H:%M}   " + ("FINISHED" if done else "still running") + "\n\n")
            for name, status, secs, rows in results:
                f.write(f"{name:<24} {status:<48} {secs / 60:6.1f} min   rows in csv: {rows}\n")

    for j in jobs:
        if os.path.exists(CACHE):                       # a kill during a save must not lose the cache
            shutil.copy(CACHE, CACHE + ".bak")
        print(f"[{datetime.datetime.now():%H:%M}] start {j['name']}", flush=True)
        status, secs = run_job(j["name"], j["cmd"], j["idle_min"] * 60, args.total_hours * 3600,
                               os.path.join(LOG_DIR, j["name"] + ".log"))
        results.append((j["name"], status, secs, count_rows(j["csv"])))
        print(f"[{datetime.datetime.now():%H:%M}] {j['name']}: {status} ({secs / 60:.1f} min)", flush=True)
        write_summary()
        time.sleep(30)                                  # let the GPU settle before the next job

    # plots and the comparison table from whatever finished
    csvs = [j["csv"] for j in jobs if j["csv"] and count_rows(j["csv"]) > 0]
    for c in csvs:
        run_job("plot", [PY, "experiments/plot_results.py", c], 300, 600,
                os.path.join(LOG_DIR, "plot_" + os.path.basename(c) + ".log"))
    existing = [r for r in ("experiments/results/alexnet_n3.csv",) if os.path.exists(r)]
    if csvs or existing:
        run_job("compare", [PY, "experiments/paper_vs_ours.py", "--sweeps"] + existing + csvs, 300, 600,
                os.path.join(LOG_DIR, "paper_vs_ours.log"))
    write_summary(done=True)
    print("[DONE] read docs/phase2/overnight_logs/summary.txt")


if __name__ == "__main__":
    main()