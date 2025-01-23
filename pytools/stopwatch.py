from __future__ import annotations

import time

from pytools import DependentDictionary, Reference


class StopWatch:
    def __init__(self) -> None:
        self.Elapsed = 0.0
        self.LastStart: float | None = None

    def start(self) -> StopWatch:
        assert self.LastStart is None

        self.LastStart = time.time()
        return self

    def stop(self) -> StopWatch:
        assert self.LastStart is not None

        self.Elapsed += time.time() - self.LastStart
        self.LastStart = None
        return self

    def elapsed(self) -> float:
        if self.LastStart:
            return time.time() - self.LastStart + self.Elapsed
        return self.Elapsed


class Job:
    def __init__(self, name: str) -> None:
        self.Name = name
        self.StopWatch = StopWatch().start()

        if self.is_visible():
            print(f"{name}...")

    def done(self) -> None:
        elapsed = self.StopWatch.elapsed()

        JOB_TIMES[self.Name] += elapsed
        if self.is_visible():
            print(" " * (len(self.Name) + 2), elapsed, "seconds")

    def is_visible(self) -> bool:
        if PRINT_JOBS.get():
            return self.Name not in HIDDEN_JOBS
        return self.Name in VISIBLE_JOBS


class EtaEstimator:
    def __init__(self, total_steps: int) -> None:
        self.stopwatch = StopWatch().start()
        self.total_steps = total_steps
        assert total_steps > 0

    def estimate(self, done: int) -> float | None:
        fraction_done = done / self.total_steps
        time_spent = self.stopwatch.elapsed()

        if fraction_done > 1.0e-5:
            return time_spent / fraction_done - time_spent
        return None


def print_job_summary() -> None:
    for key, value in JOB_TIMES.iteritems():
        print(key, " " * (50 - len(key)), value)


HIDDEN_JOBS: list[str] = []
VISIBLE_JOBS: list[str] = []
JOB_TIMES = DependentDictionary(lambda x: 0)
PRINT_JOBS = Reference(True)
