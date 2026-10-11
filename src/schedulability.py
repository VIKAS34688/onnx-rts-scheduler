"""
Response-time analysis and splitting optimization.

Direct re-implementation of the reference paper's Section VI (Equations
1-4) and Section VII (Algorithm 1 - optimal, Algorithm 2 - greedy
heuristic). No new scheduling theory is introduced here.

TIME UNIT NOTE: the paper's analysis is discrete-time, so the "-1" in
B_i = max{C_j^max - 1} (and in the h*T_j - 1 points of Eq. 4) means
"one time unit". We measure in milliseconds, so one unit must be a tiny
tick (1 microsecond), NOT a full 1.0 ms -- subtracting a whole
millisecond would shrink the blocking term and make the analysis
optimistic (unsafe).

BETA NOTE: Eq. (3) as printed sums over tau_h with h > i, but the paper
itself says beta_i depends only on tau_i and HIGHER-priority tasks, and
the formula only makes sense (slack = deadline - own cost - higher-
priority interference) with higher-priority tasks. We implement the
higher-priority version (h < i) and cross-check it against
response_time() in the __main__ block, instead of trusting the printed
symbol.

Priority convention: lower priority NUMBER = higher priority.
"""

import math

TICK_MS = 0.001  # one discrete time unit = 1 microsecond


class RTSTask:
    """chunk_wcets_ms: chunk WCETs in execution order (ms). One element = unsplit."""

    def __init__(self, name, priority, period_ms, deadline_ms, chunk_wcets_ms):
        self.name = name
        self.priority = priority
        self.period_ms = period_ms
        self.deadline_ms = deadline_ms
        self.chunk_wcets_ms = list(chunk_wcets_ms)

    @property
    def C(self):
        return sum(self.chunk_wcets_ms)

    @property
    def C_max(self):
        return max(self.chunk_wcets_ms)

    @property
    def C_last(self):
        return self.chunk_wcets_ms[-1]

    def __repr__(self):
        return (f"RTSTask({self.name}, prio={self.priority}, "
                f"C={self.C:.3f}ms, chunks={len(self.chunk_wcets_ms)})")


def _higher_or_equal(all_tasks, task):
    return [t for t in all_tasks if t.priority <= task.priority]


def _strictly_higher(all_tasks, task):
    return [t for t in all_tasks if t.priority < task.priority]


def _strictly_lower(all_tasks, task):
    return [t for t in all_tasks if t.priority > task.priority]


def blocking_factor(task, all_tasks) -> float:
    """B_i = max over lower-priority tau_j of (C_j^max - 1 tick)."""
    lower = _strictly_lower(all_tasks, task)
    if not lower:
        return 0.0
    return max(t.C_max - TICK_MS for t in lower)


def busy_period(task, all_tasks, include_blocking=True, max_iters=1000, tol=1e-9):
    """Eq. (1): level-i busy period I_i and K_i = ceil(I_i / T_i)."""
    B_i = blocking_factor(task, all_tasks) if include_blocking else 0.0
    heq = _higher_or_equal(all_tasks, task)

    I = B_i + task.C
    for _ in range(max_iters):
        I_next = B_i + sum(math.ceil(I / t.period_ms) * t.C for t in heq)
        if abs(I_next - I) < tol:
            I = I_next
            break
        I = I_next
    return I, math.ceil(I / task.period_ms)


def response_time(task, all_tasks, max_iters=1000, tol=1e-9) -> float:
    """Eq. (2): worst-case response time over the K_i jobs in the busy period."""
    B_i = blocking_factor(task, all_tasks)
    higher = _strictly_higher(all_tasks, task)
    _, K_i = busy_period(task, all_tasks)

    worst_R = 0.0
    for k in range(1, K_i + 1):
        s = B_i + task.C - task.C_last + sum(t.C for t in higher)
        for _ in range(max_iters):
            interference = sum((math.floor(s / t.period_ms) + 1) * t.C for t in higher)
            s_next = B_i + k * task.C - task.C_last + interference
            if abs(s_next - s) < tol:
                s = s_next
                break
            s = s_next
        worst_R = max(worst_R, s + task.C_last - (k - 1) * task.period_ms)
    return worst_R


def is_schedulable(task, all_tasks) -> bool:
    return response_time(task, all_tasks) <= task.deadline_ms


def blocking_tolerance(task, all_tasks) -> float:
    """
    Eq. (3)-(4): beta_i = largest non-preemptive blocking tau_i can absorb
    without missing its deadline. Depends only on tau_i and higher-priority
    tasks (the paper says so explicitly), so the busy period used for K_i
    excludes blocking.
    """
    higher = _strictly_higher(all_tasks, task)
    heq = higher + [task]
    _, K_i = busy_period(task, heq, include_blocking=False)

    best_beta = float("inf")
    for k in range(1, K_i + 1):
        lo = (k - 1) * task.period_ms
        hi = (k - 1) * task.period_ms + task.deadline_ms - task.C_last

        points = {hi}
        for t in heq:
            h = 1
            while h * t.period_ms - TICK_MS <= hi:
                p = h * t.period_ms - TICK_MS
                if p > lo:
                    points.add(p)
                h += 1

        beta_k = max(
            t_val - k * task.C + task.C_last
            - sum((math.floor(t_val / t.period_ms) + 1) * t.C for t in higher)
            for t_val in points
        )
        best_beta = min(best_beta, beta_k)
    return best_beta


def min_blocking_budget(higher_priority_tasks) -> float:
    """B in Algorithms 1/2: the smallest beta among higher-priority tasks."""
    if not higher_priority_tasks:
        return float("inf")
    return min(
        blocking_tolerance(t, [h for h in higher_priority_tasks if h.priority <= t.priority])
        for t in higher_priority_tasks
    )


# ---------------------------------------------------------------------
# Splitting optimization (Section VII, Algorithms 1 and 2)
# ---------------------------------------------------------------------

SEARCH_RUNS = 300       # profiling runs per chunk inside the search
WCET_STAT = "p99_ms"    # estimator used as a chunk's WCET ("max_ms" = the paper's observed max)


def _profile_config(model_path, active_points, sample_input, cache, tmp_dir):
    """Build + profile one split configuration; cached like the paper's profiling cache."""
    import splitter
    import profiler

    key = tuple(active_points)
    if key in cache:
        return cache[key]
    chunk_paths = splitter.build_chunks(model_path, list(active_points), tmp_dir)
    wcets = [profiler.profile_chunk(p, sample_input, n_runs=SEARCH_RUNS)[WCET_STAT]
             for p in chunk_paths]
    cache[key] = wcets
    return wcets


def optimal_split(task_name, priority, period_ms, deadline_ms, model_path,
                  candidate_points, higher_priority_tasks, sample_input,
                  tmp_dir="models/chunks_opt"):
    """
    Algorithm 1. WARNING: worst case is 2^(number of candidate points)
    configurations -- the paper itself restricted itself to 5-11 points.
    Pass a SHORT candidate list (e.g. 5-8 block boundaries), not all 19.
    """
    cache = {}
    B = min_blocking_budget(higher_priority_tasks)

    full = _profile_config(model_path, candidate_points, sample_input, cache, tmp_dir)
    if B < max(full):
        return None  # even fully split is not schedulable

    pending, seen, valid = [tuple()], {tuple()}, []
    while pending:
        active = pending.pop(0)
        wcets = _profile_config(model_path, active, sample_input, cache, tmp_dir)
        if B < max(wcets):
            for p in candidate_points:
                if p not in active:
                    cfg = tuple(sorted(active + (p,), key=candidate_points.index))
                    if cfg not in seen:
                        seen.add(cfg)
                        pending.append(cfg)
        else:
            valid.append((active, sum(wcets), wcets))

    if not valid:
        return None
    best_active, _, best = min(valid, key=lambda x: x[1])
    result = RTSTask(task_name, priority, period_ms, deadline_ms, best)
    result.active_points = best_active
    return result


def greedy_split(task_name, priority, period_ms, deadline_ms, model_path,
                 candidate_points, higher_priority_tasks, sample_input,
                 tmp_dir="models/chunks_greedy"):
    """Algorithm 2: keep only the single best candidate per iteration."""
    cache = {}
    B = min_blocking_budget(higher_priority_tasks)

    full = _profile_config(model_path, candidate_points, sample_input, cache, tmp_dir)
    if B < max(full):
        return None

    active = tuple()
    wcets = _profile_config(model_path, active, sample_input, cache, tmp_dir)
    while B < max(wcets):
        best = None
        for p in candidate_points:
            if p not in active:
                trial = tuple(sorted(active + (p,), key=candidate_points.index))
                tw = _profile_config(model_path, trial, sample_input, cache, tmp_dir)
                if best is None or max(tw) < max(best[1]):
                    best = (trial, tw)
        active, wcets = best
    result = RTSTask(task_name, priority, period_ms, deadline_ms, wcets)
    result.active_points = active
    return result


if __name__ == "__main__":
    ALEXNET_SPLIT = [0.182, 0.135, 0.085, 0.216, 0.288, 0.077, 0.137, 0.192, 0.161, 0.037, 3.292]

    print("=== Check 1: beta vs response_time agree ===")
    tau1 = RTSTask("tau1", 1, 5, 5, [1.5])
    beta = blocking_tolerance(tau1, [tau1])
    print(f"beta(tau1) = {beta:.3f}ms  (expected D - C = {5 - 1.5:.3f}ms)")
    for c in [beta - 0.01, beta, beta + 0.001, beta + 0.01]:
        low = RTSTask("low", 2, 50, 50, [c])
        R = response_time(tau1, [tau1, low])
        print(f"  lower-priority chunk = {c:.3f}ms -> R1 = {R:.3f}ms  schedulable: {R <= 5}")

    print("\n=== Check 2: AlexNet as the lower-priority task ===")
    for label, chunks in [("unsplit (4.469ms)", [4.469]),
                          ("fully split (max 3.292ms)", ALEXNET_SPLIT)]:
        tau2 = RTSTask("tau2", 2, 50, 50, chunks)
        R = response_time(tau1, [tau1, tau2])
        print(f"  {label}: R1 = {R:.3f}ms  schedulable: {R <= 5}")