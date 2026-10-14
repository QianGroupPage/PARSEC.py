"""Self-consistent spin-orbit coupling through the CuPy spinor backend.

Correctness-first baseline: builds a fresh
:class:`~parsec_python.acceleration.backends.cupy_spinor.CuPySpinorHamiltonian`
from the existing CuPy scalar backend's already-uploaded host kinetic/
nonlocal arrays every SCF iteration (re-uploading them as complex128 each
time -- the production real-valued scalar backend instead uploads these
once and only mutates the local potential afterward; matching that here is
explicit follow-up optimization work, not a correctness requirement), and
solves with
:func:`~parsec_python.acceleration.Eigensolvers.chebff_spinor.cupy_spinor_eigenproblem_solver`.
Everything else -- density, Hartree, XC, Anderson mixing, total energy --
stays on the host exactly as in the reference solver, the same division of
labor
:mod:`~parsec_python.acceleration.SCF.single_point` already uses for the
scalar path (GPU for the expensive Hamiltonian-apply/eigensolve, CPU for the
cheap per-iteration bookkeeping).
"""

from __future__ import annotations

from typing import Callable

from parsec_python.Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from parsec_python.SCF.self_consistent_soc import (
    run_self_consistent_soc as run_reference_self_consistent_soc,
)
from parsec_python.models import SelfConsistentSOCIteration, SelfConsistentSOCResult

from ..backends.cupy_spinor import CuPySpinorHamiltonian
from ..Eigensolvers.chebff_spinor import cupy_spinor_eigenproblem_solver
from .single_point import AcceleratedPreparedSinglePointSystem


def _cupy_spinor_hamiltonian_builder(scalar_hamiltonian, soc_projectors):
    """Read the current iteration's uploaded host arrays back off the
    scalar CuPy backend instead of re-deriving them.

    ``scalar_hamiltonian`` is the ``CuPyBoundHamiltonian`` returned by
    ``system.hamiltonian(input_potential)``; its ``.backend`` (a
    ``CuPyHamiltonianBackend``) already has ``host_negative_laplacian``/
    ``host_nonlocal_operator`` uploaded once at system preparation and
    ``local_potential`` set to the current ``input_potential`` by the
    ``update_local`` call ``CuPyBoundHamiltonian.__init__`` just made.
    """

    backend = scalar_hamiltonian.backend
    return CuPySpinorHamiltonian(
        backend.host_negative_laplacian,
        backend.local_potential,
        backend.host_nonlocal_operator,
        soc_projectors,
    )


def run_self_consistent_soc(
    system: AcceleratedPreparedSinglePointSystem,
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...],
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Run self-consistent SOC with the CuPy spinor Hamiltonian/eigensolver.

    Requires ``system.backend_info.selected == "cupy"``: the scipy/native
    backends don't build a ``CuPyHamiltonianBackend`` to read host arrays
    from, and periodic systems aren't supported by this Gamma-point-only
    tier (see ``acceleration.Hamiltonian.kpoint_spinor_operator`` -- not yet
    implemented -- for the periodic follow-up).
    """

    if system.backend_info.selected != "cupy":
        raise ValueError(
            "accelerated run_self_consistent_soc requires a CuPy-bound "
            f"system, got backend={system.backend_info.selected!r}"
        )
    return run_reference_self_consistent_soc(
        system,
        soc_projectors,
        callback=callback,
        eigenproblem_solver=cupy_spinor_eigenproblem_solver,
        spinor_hamiltonian_builder=_cupy_spinor_hamiltonian_builder,
    )


__all__ = ["run_self_consistent_soc"]
