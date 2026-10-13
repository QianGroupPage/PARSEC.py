"""Open-boundary Poisson solves of orbital pair densities.

The exact exchange of a post-SCF energy refinement
(:mod:`parsec_python.V_xc.exact_exchange`) needs the potential of every
product of two occupied orbitals.  This module provides those solves from
the Hartree components a prepared accelerated system already has; it defines
no boundary model and no linear solver of its own.

A pair density differs from the SCF density in two ways that decide which
components can serve:

* it has no symmetry of the molecule, so the solve runs on the full grid,
  never on the wedge of a symmetry-reduced Hartree solver;
* its boundary values are its own multipole expansion (sphere) or direct
  Coulomb sum (box).  The atomic tail a system may add to the boundary of
  the total density is a static term of that density and is left out.

The conjugate gradients start from zero for every pair: the chronological
starting vector of the SCF Hartree solver extrapolates a sequence of similar
densities, which successive pairs are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parsec_python.V_xc.exact_exchange import pair_hartree_settings

from ..models import BackendUnavailableError
from .poisson import build_hartree_problem, solve_scipy_hartree


@dataclass(frozen=True)
class PairPotential:
    """Potential of one pair density in Rydberg and its CG iteration count."""

    potential: np.ndarray
    iterations: int


class PairPoissonSolver:
    """Full-grid Poisson solver of pair densities for one prepared system.

    ``right_hand_side(density)`` returns the boundary-corrected source and
    ``conjugate_gradient`` is a prepared CG backend with the ``solve``
    contract of
    :class:`parsec_python.acceleration.backends.native.NativeConjugateGradientBackend`.
    ``description`` is one line for the report.
    """

    def __init__(self, size, settings, right_hand_side, conjugate_gradient, description):
        self.size = int(size)
        self.settings = settings
        self._right_hand_side = right_hand_side
        self._conjugate_gradient = conjugate_gradient
        self.description = description

    def __call__(self, pair_density: np.ndarray) -> PairPotential:
        density = np.ascontiguousarray(pair_density, dtype=np.float64)
        if density.shape != (self.size,):
            raise ValueError("pair density does not match the active grid")
        payload = self._conjugate_gradient.solve(
            self._right_hand_side(density),
            np.zeros(self.size, dtype=np.float64),
            relative_tolerance=self.settings.relative_tolerance,
            absolute_tolerance=self.settings.absolute_tolerance,
            max_iterations=self.settings.max_iterations,
        )
        if not payload["converged"]:
            raise RuntimeError(
                "pair-density conjugate-gradient solve did not converge: "
                f"residual={float(payload['residual_norm']):.3e}, "
                f"matvecs={int(payload['matrix_vector_products'])}"
            )
        return PairPotential(
            potential=np.asarray(payload["solution"], dtype=np.float64),
            iterations=int(payload["iterations"]),
        )


class _CallablePairSolver:
    """A pair solver made of one complete Hartree solve per pair."""

    def __init__(self, solve, description):
        self._solve = solve
        self.description = description

    def __call__(self, pair_density):
        return self._solve(pair_density)


def _boundary_method(grid, settings) -> str:
    method = settings.boundary_method
    if method == "auto":
        method = "multipole" if grid.settings.domain_shape == "sphere" else "direct"
    return method


def _right_hand_side_builder(grid, settings):
    """Return ``(density -> right-hand side, description)`` for pair densities."""

    method = _boundary_method(grid, settings)
    if method == "multipole" and grid.settings.domain_shape == "sphere":
        try:
            from .native_boundary import NativeMultipoleBoundaryBuilder

            # A new builder: the one of the SCF may hold the atomic tail and
            # may work on the symmetry wedge.
            builder = NativeMultipoleBoundaryBuilder(grid, settings.multipole_order)
        except (BackendUnavailableError, RuntimeError, ValueError):
            builder = None
        if builder is not None:
            return (
                lambda density: builder.build(density)[0],
                f"C++/OpenMP multipole boundary of order {settings.multipole_order}",
            )

    def host_right_hand_side(density):
        return build_hartree_problem(density, grid, settings, None)[0]

    return host_right_hand_side, (
        f"multipole boundary of order {settings.multipole_order}"
        if method == "multipole"
        else "direct Coulomb-sum boundary"
    )


def _prepared_conjugate_gradient(system, reference):
    """Return ``(CG backend, description)`` on the full grid, or ``None``.

    The backend of the SCF Hartree solver is reused where that solver works
    on the full grid.  Where it works on a symmetry wedge a full-grid backend
    of the same kind (C++/OpenMP, or CUDA where the SCF solve ran there) is
    prepared from the finite-difference Laplacian.
    """

    from ..Laplacian import materialize_negative_laplacian

    size = int(reference.grid.size)
    existing = getattr(system.backend, "native_poisson_solver", None)
    inner = getattr(existing, "solver", existing)
    backend = getattr(inner, "backend", None)
    on_device = type(backend).__name__ == "CuPyPreparedConjugateGradientBackend"

    def describe(backend, origin):
        if type(backend).__name__ == "CuPyPreparedConjugateGradientBackend":
            return f"FP64 CUDA conjugate gradients on device {backend.device_id} ({origin})"
        workers = int(backend.worker_count)
        return (
            f"C++/OpenMP conjugate gradients, {workers} "
            f"thread{'' if workers == 1 else 's'} ({origin})"
        )

    reduced = existing is not None and hasattr(existing, "reduction")
    if (
        backend is not None
        and not reduced
        and tuple(int(value) for value in backend.shape) == (size, size)
    ):
        return backend, describe(backend, "the solver of the SCF Hartree potential")
    try:
        operator = materialize_negative_laplacian(reference.negative_laplacian)
        if on_device:
            from .cupy_prepared import CuPyPreparedConjugateGradientBackend

            backend = CuPyPreparedConjugateGradientBackend(
                operator, device_id=backend.device_id
            )
        else:
            from ..backends.native import NativeConjugateGradientBackend

            backend = NativeConjugateGradientBackend(operator)
    except BackendUnavailableError:
        return None
    return backend, describe(backend, "prepared for the full grid")


def _boundary_description(grid, settings) -> str:
    if _boundary_method(grid, settings) == "multipole":
        return f"multipole boundary of order {settings.multipole_order}"
    return "direct Coulomb-sum boundary"


def build_pair_poisson(system):
    """Return the pair-density Poisson solver of a prepared accelerated system.

    The result is callable as
    :func:`parsec_python.V_xc.exact_exchange.exact_exchange_energy` expects
    and carries a ``description`` for the report.  It follows the Hartree
    backend the system was prepared with: an explicit SciPy calculation
    stays on SciPy, the CuPy Hartree backend solves on its device, and the
    native backend uses its conjugate gradients, on the GPU where the SCF
    ran them there.
    """

    reference = getattr(system, "reference", system)
    grid = reference.grid
    settings = pair_hartree_settings(reference.input.hartree)
    selected = str(
        dict(getattr(system.backend_info, "details", ())).get("hartree_backend", "")
    )

    def host_solver():
        # SciPy conjugate gradients with the fast host multipoles of the
        # SciPy Hartree backend.
        def solve(density):
            return solve_scipy_hartree(
                density, grid, reference.negative_laplacian, settings
            )

        return _CallablePairSolver(
            solve,
            "SciPy conjugate gradients on the full grid; "
            + _boundary_description(grid, settings),
        )

    if selected == "scipy":
        return host_solver()

    # The CuPy Hartree backend: its solver works on the full grid and builds
    # the boundary of whatever density it is given.
    device_solver = getattr(system.backend, "poisson_solver", None)
    if device_solver is not None and callable(getattr(device_solver, "solve", None)):
        return _CallablePairSolver(
            lambda density: device_solver.solve(density, grid, settings, None),
            "CuPy conjugate gradients on the full grid (the solver of the SCF "
            "Hartree potential); " + _boundary_description(grid, settings),
        )

    prepared = _prepared_conjugate_gradient(system, reference)
    if prepared is None:
        # No compiled extension.
        return host_solver()
    backend, linear = prepared
    right_hand_side, boundary = _right_hand_side_builder(grid, settings)
    return PairPoissonSolver(
        grid.size, settings, right_hand_side, backend, f"{linear}; {boundary}"
    )


__all__ = ["PairPoissonSolver", "PairPotential", "build_pair_poisson"]
