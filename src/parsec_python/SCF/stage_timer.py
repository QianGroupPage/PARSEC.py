"""Seconds per SCF stage, shared by the drivers that report a timing table."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable


class StageTimer:
    """Seconds per named stage; ``synchronize`` (a device barrier) makes them exact on a GPU.

    ``seconds`` accumulates over the run; ``lap()`` returns the seconds added to each stage since
    the previous call, which gives the per-iteration split.
    """

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self.seconds: dict[str, float] = {}
        self._synchronize = synchronize
        self._lap_start: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if self._synchronize is not None:
            self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._synchronize is not None:
                self._synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def add_counters(self, prefix: str, counters) -> None:
        for name, value in (counters or {}).items():
            key = f"{prefix}{name}"
            self.seconds[key] = self.seconds.get(key, 0.0) + float(value)

    def lap(self) -> dict[str, float]:
        """Seconds each stage gained since the previous ``lap()``."""
        gained = {
            name: value - self._lap_start.get(name, 0.0) for name, value in self.seconds.items()
        }
        self._lap_start = dict(self.seconds)
        return gained


__all__ = ["StageTimer"]
