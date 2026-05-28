"""Thread-safe session-scoped benchmark result collector."""

from __future__ import annotations

import threading


class BenchmarkCollector:
    """Collects benchmark results across a pytest session.

    Not a process-level singleton — held as a session-scoped fixture.
    Thread-safe for safety, though benchmarks run serially (-p no:xdist).
    """

    def __init__(self):
        self._results: list[dict] = []
        self._lock = threading.Lock()

    def add_result(self, case: dict) -> None:
        with self._lock:
            self._results.append(case)

    def get_results(self) -> list[dict]:
        with self._lock:
            return list(self._results)

    def clear(self) -> None:
        with self._lock:
            self._results.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)
