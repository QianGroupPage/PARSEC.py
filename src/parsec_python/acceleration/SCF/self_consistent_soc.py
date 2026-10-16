"""Self-consistent spin-orbit coupling of an isolated system on the GPU (Gamma point).

The reference loop of :mod:`parsec_python.SCF.self_consistent_soc` with three
device pieces plugged in:

* the spinor Hamiltonian is the device operator of
  :mod:`parsec_python.acceleration.Hamiltonian.device_bloch` (real sparse
  products on the real view of the complex block, the spin-orbit term of all
  atoms as two sparse products), with its static matrices uploaded once and only
  the local potential uploaded per SCF iteration;
* the eigensolver is :func:`~parsec_python.acceleration.Eigensolvers.chebff_spinor.solve_spinor_eigval`:
  CHEBFF in the first iteration and one SUBSPACE step in every later one, as the
  reference ``solve_eigval`` does, with the orbitals resident on the device;
* the density is formed on the device from those orbitals.

Hartree, exchange-correlation, mixing and the energies stay on the host exactly
as in the reference, as in :mod:`parsec_python.acceleration.SCF.single_point`.
The orbitals come to the host once, at the end.

``backend`` selects the array backend: ``None`` is the current CUDA device through
CuPy; the tests pass a :class:`NumpyArrayBackend` to run this code on the host.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from parsec_python.Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from parsec_python.SCF.self_consistent_soc import (
    run_self_consistent_soc as run_reference_self_consistent_soc,
)
from parsec_python.models import SelfConsistentSOCIteration, SelfConsistentSOCResult

from ..Eigensolvers.chebff_spinor import solve_spinor_eigval, spinor_density_on_device
from ..Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
from .single_point import AcceleratedPreparedSinglePointSystem


class _HostScalar:
    """Host-side description of the scalar Gamma-point Hamiltonian, as the converter reads it."""

    _k_gradient = None

    def __init__(self, negative_laplacian, nonlocal_operator, potential) -> None:
        self.negative_laplacian = negative_laplacian
        self.nonlocal_operator = nonlocal_operator
        self._diagonal = potential


class _HostSpinor:
    xc_delta = None
    xc_field = None

    def __init__(self, scalar: _HostScalar, soc_projectors) -> None:
        self.scalar_hamiltonian = scalar
        self.soc_projectors = soc_projectors


def _host_parts(scalar_hamiltonian):
    """Kinetic matrix, nonlocal operator and local potential of a bound scalar Hamiltonian."""
    backend = getattr(scalar_hamiltonian, "backend", None)
    if backend is not None and hasattr(backend, "host_negative_laplacian"):
        return (
            backend.host_negative_laplacian,
            backend.host_nonlocal_operator,
            np.asarray(backend.local_potential, dtype=np.float64),
        )
    return (
        scalar_hamiltonian.negative_laplacian,
        scalar_hamiltonian.nonlocal_operator,
        np.asarray(scalar_hamiltonian.effective_potential, dtype=np.float64),
    )


class DeviceSpinorHamiltonianBuilder:
    """``spinor_hamiltonian_builder`` hook: the device operator of one SCF iteration.

    The converter caches the uploaded Laplacian, projectors and spin-orbit
    matrices by the identity of the host objects, which persist across the
    iterations, so an iteration uploads only its local potential.
    """

    def __init__(self, backend) -> None:
        self.converter = DeviceBlochConverter(backend)

    def __call__(self, scalar_hamiltonian, soc_projectors):
        laplacian, nonlocal_operator, potential = _host_parts(scalar_hamiltonian)
        return self.converter.operator(
            _HostSpinor(_HostScalar(laplacian, nonlocal_operator, potential), soc_projectors)
        )


def run_self_consistent_soc(
    system: AcceleratedPreparedSinglePointSystem,
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...],
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    backend=None,
) -> SelfConsistentSOCResult:
    """Run self-consistent SOC with the device spinor Hamiltonian and eigensolver.

    Without ``backend`` the system must be bound to CuPy
    (``system.backend_info.selected == "cupy"``); periodic systems are not
    covered by this Gamma-point tier.
    """
    if backend is None:
        if system.backend_info.selected != "cupy":
            raise ValueError(
                "accelerated run_self_consistent_soc requires a CuPy-bound "
                f"system, got backend={system.backend_info.selected!r}"
            )
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        backend = CupyArrayBackend(int(cp.cuda.Device().id))
    backend.use()
    xp = backend.xp

    def solver(operator, requested_states, *, settings, state=None):
        return solve_spinor_eigval(
            operator,
            requested_states,
            settings=settings,
            state=state,
            xp=xp,
            sync=backend.synchronize,
        )

    return run_reference_self_consistent_soc(
        system,
        soc_projectors,
        callback=callback,
        eigenproblem_solver=solver,
        spinor_hamiltonian_builder=DeviceSpinorHamiltonianBuilder(backend),
        orbital_density_builder=lambda spinors, occupations, volume_element: spinor_density_on_device(
            xp, spinors, occupations, volume_element
        ),
        synchronize=backend.synchronize,
        to_host=backend.asnumpy,
    )


__all__ = ["DeviceSpinorHamiltonianBuilder", "run_self_consistent_soc"]
