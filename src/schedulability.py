"""
Response-time analysis and splitting optimization.

Direct re-implementation of the reference paper's Section VI (Equations
1-4) and Section VII (Algorithm 1 - optimal, Algorithm 2 - greedy
heuristic). No new scheduling theory is introduced here — this module's
job is to faithfully reproduce their math against our own profiled WCETs.

Notation, matching the paper:
    C_i,j   — WCET of chunk j of task i
    C_i     — total WCET of task i (sum of its chunks)
    B_i     — blocking factor for task i
    beta_i  — blocking tolerance for task i

TODO (Person B):
- response_time_analysis(): implement Eq. (1) and (2) recurrence relations
- blocking_tolerance(): implement Eq. (3) and (4)
- optimal_split(): Algorithm 1 — explore configurations, keep the lowest
  total-WCET schedulable one
- greedy_split(): Algorithm 2 — refine one candidate configuration at a
  time, faster but potentially suboptimal
"""


def response_time_analysis(tasks: list) -> dict:
    """
    Given a list of tasks (each with priority, period, deadline, and a
    list of chunk WCETs), compute the worst-case response time for each
    task per the paper's Eq. (1)-(2) recurrence.

    TODO: implement.
    """
    raise NotImplementedError


def blocking_tolerance(tasks: list) -> dict:
    """
    Compute beta_i for each task per the paper's Eq. (3)-(4).

    TODO: implement.
    """
    raise NotImplementedError


def optimal_split(task, candidate_split_points, profile_fn):
    """
    Algorithm 1 from the paper: iteratively activate split points,
    profiling each candidate configuration, keeping every configuration
    that satisfies the blocking tolerance, and returning the one with
    the lowest total summed WCET.

    TODO: implement the loop described in Algorithm 1 (start from
    zero splits, escalate only when the current config violates the
    blocking tolerance from higher-priority tasks).
    """
    raise NotImplementedError


def greedy_split(task, candidate_split_points, profile_fn):
    """
    Algorithm 2 from the paper: same idea as optimal_split, but only
    keeps one best candidate per iteration instead of exploring the
    full configuration space.

    TODO: implement.
    """
    raise NotImplementedError
