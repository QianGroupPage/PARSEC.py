"""Readable reference workflow for one isolated real-space calculation.

The numerical implementation lives in :mod:`parsec_python.SCF`.  The public
package API defaults to :mod:`parsec_python.acceleration.driver`; this module
remains the inspectable scientific reference and fallback implementation.
"""

from __future__ import annotations

from typing import Callable

from .SCF.pbc import (
    PeriodicPreparedSinglePointSystem,
    prepare_periodic_single_point as _prepare_periodic_single_point,
    #run_periodic_single_point as _run_periodic_single_point,
)
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


# Periodic (Gamma-point) counterparts.  See SCF/pbc.py's module docstring
# for what the periodic path does and does not yet model correctly.
def prepare_periodic_single_point(
    problem: SinglePointInput,
) -> PeriodicPreparedSinglePointSystem:
    """Prepare an inspectable periodic system without entering the SCF loop."""
    return _prepare_periodic_single_point(problem)


#def run_periodic_single_point(
#    problem: SinglePointInput,
#    *,
#    callback: Callable[[SCFIteration], None] | None = None,
#) -> SinglePointResult:
#    """Run the complete two-stage Gamma-point periodic single-point workflow."""
#    return _run_periodic_single_point(problem, callback=callback)


__all__ = [
    "PreparedSinglePointSystem",
    "prepare_single_point",
    "run_scf",
    "run_single_point",
    "PeriodicPreparedSinglePointSystem",
    "prepare_periodic_single_point",
#    "run_periodic_single_point",
]
