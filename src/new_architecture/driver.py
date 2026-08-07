"""Readable top-level workflow for one isolated CA-LDA calculation.

The numerical implementation lives in :mod:`new_architecture.SCF`.
This module only makes the two top-level stages explicit and composes them.
"""

from __future__ import annotations

from typing import Callable

from .SCF.single_point import (
    PreparedSinglePointSystem,
    prepare_single_point as _prepare_single_point,
    run_scf as _run_scf,
)
from .models import SCFIteration, SinglePointInput, SinglePointResult


# Stage 1: construct the grid and every potential-independent component.
def prepare_single_point(problem: SinglePointInput) -> PreparedSinglePointSystem:
    """Prepare an inspectable system without entering the SCF loop."""
    return _prepare_single_point(problem)


# Stage 2: iterate density, potentials, eigenstates, and occupations to SCF.
def run_scf(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
) -> SinglePointResult:
    """Run the SCF iteration for an already prepared system."""
    return _run_scf(system, callback=callback)


def run_single_point(
    problem: SinglePointInput,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
) -> SinglePointResult:
    """Run the complete two-stage isolated single-point workflow."""
    prepared_system = prepare_single_point(problem)
    return run_scf(prepared_system, callback=callback)

__all__ = [
    "PreparedSinglePointSystem",
    "prepare_single_point",
    "run_scf",
    "run_single_point",
]
