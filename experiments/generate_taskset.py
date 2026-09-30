"""
UUniFast algorithm for generating random task utilizations, matching the
paper's own experimental method (Section VIII-A, Bini & Buttazzo 2005).

Given a target total utilization U and a number of tasks n, produces n
individual utilizations U_1..U_n that sum to U, avoiding the bias of a
naive uniform split.

TODO (Person C):
- Implement uunifast(n, U) -> list[float]
- Implement generate_task_set(n, U, period_range) -> list[Task], assigning
  periods (and Deadline-Monotonic priorities from those periods) alongside
  the generated utilizations
"""


def uunifast(n: int, total_utilization: float) -> list:
    """
    TODO: implement the standard UUniFast algorithm.
    """
    raise NotImplementedError


def generate_task_set(n: int, total_utilization: float, period_range: tuple):
    """
    TODO: combine uunifast() with randomly sampled periods to build a
    full task set, then assign priorities via Deadline-Monotonic
    (shorter deadline/period = higher priority).
    """
    raise NotImplementedError
