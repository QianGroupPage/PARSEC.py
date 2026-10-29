"""The operations an ionic loop needs from an execution path.

A relaxation and a molecular dynamics run repeat the same four operations
for every structure: solve the electronic structure, evaluate the forces,
rebuild what depends on the atom positions, and hand the next SCF loop what
it continues from.  How they are executed is the business of the readable
NumPy path or of an accelerated path; an :class:`IonicBackend` names them so
that the loops in :mod:`parsec_python.Relaxation` and
:mod:`parsec_python.MolecularDynamics` are written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..models import Atom


@dataclass(frozen=True)
class IonicBackend:
    """The four operations of one execution path.

    ``run_scf(system, callback=..., warm_start=...)`` returns the SCF result
    of a prepared system; ``compute_forces(system, result, symmetrize=...,
    operations=...)`` returns a :class:`parsec_python.Forces.ForceResult`;
    ``update_geometry(system, atoms)`` returns the prepared system of the
    moved atoms on the same grid; ``warm_start(previous_system,
    previous_result, system)`` returns what the next SCF loop takes over, in
    the form ``run_scf`` accepts.  ``restart_scf(system, result)``, where
    given, is what a second SCF loop of the same structure continues from,
    without any extrapolation (``Relaxation_SCF_Upscale``); without it the
    loop takes ``warm_start(system, result, system)``.  ``run_scf`` takes
    ``convergence_criterion=`` where the relaxation loosens the criterion.
    """

    run_scf: Callable[..., object]
    compute_forces: Callable[..., object]
    update_geometry: Callable[[object, Sequence[Atom]], object]
    warm_start: Callable[[object, object, object], object]
    restart_scf: Callable[[object, object], object] | None = None


def reference_backend(potential_extrapolation: str = "none") -> IonicBackend:
    """The readable full-grid NumPy/SciPy path.

    ``potential_extrapolation`` is how the SCF loop of moved atoms starts
    (``Potential_Extrapolation``; see
    :func:`parsec_python.SCF.warm_start_for_moved_atoms`).
    """

    from functools import partial

    from ..Forces import compute_forces
    from ..SCF import run_scf, update_geometry, warm_start_for_moved_atoms
    from ..SCF.single_point import extrapolation_keywords

    keywords = extrapolation_keywords(potential_extrapolation)
    return IonicBackend(
        run_scf=run_scf,
        compute_forces=compute_forces,
        update_geometry=update_geometry,
        warm_start=partial(warm_start_for_moved_atoms, **keywords),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


def periodic_backend(
    k_points=None, k_weights=None, potential_extrapolation: str = "none"
) -> IonicBackend:
    """The readable path for a periodic cell: the Gamma point, or a uniform k-point sampling.

    ``k_points`` and ``k_weights`` (the Monkhorst-Pack points of the input, equal weights) select the k-point driver
    :func:`parsec_python.SCF.kpoints.run_scf_kpoints`, which keeps no eigenvectors, so its SCF step also accumulates
    the nonlocal force from them; without them the Gamma-point loop runs and the forces are built from its orbitals.
    The forces are those of :func:`parsec_python.Forces.compute_periodic_forces`: no symmetrization (the structure is
    not reduced by space-group operations) and the net force removed, which is what Fortran does for a crystal.  The
    k-point loops start their eigenproblems cold after a move; the Gamma loop carries its eigensubspace over.
    """

    from functools import partial

    import numpy as np

    from ..Forces import compute_periodic_forces
    from ..SCF import run_scf, update_periodic_geometry, warm_start_for_moved_atoms
    from ..SCF.kpoints import run_scf_kpoints
    from ..SCF.single_point import extrapolation_keywords

    sampled = k_points is not None
    if sampled:
        k_points = np.asarray(k_points, dtype=float)
        k_weights = np.asarray(k_weights, dtype=float)

    def solve(system, *, callback=None, warm_start=None, convergence_criterion=None):
        if sampled:
            return run_scf_kpoints(
                system,
                k_points,
                k_weights,
                callback=callback,
                compute_forces=True,
                warm_start=warm_start,
                convergence_criterion=convergence_criterion,
            )
        return run_scf(
            system, callback=callback, warm_start=warm_start, convergence_criterion=convergence_criterion
        )

    def forces(system, result, *, symmetrize=False, operations=None):
        if symmetrize or operations is not None:
            raise ValueError("a periodic cell has no point-group symmetrization of its forces here")
        return compute_periodic_forces(system, result)

    keywords = extrapolation_keywords(potential_extrapolation)
    return IonicBackend(
        run_scf=solve,
        compute_forces=forces,
        update_geometry=update_periodic_geometry,
        warm_start=partial(warm_start_for_moved_atoms, **keywords),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


def release_orbitals(result) -> None:
    """Drop the views a finished result holds on the eigensolver subspace.

    An accelerated eigensolver keeps the subspace the next SCF loop continues
    from; views of it in the result of the previous structure would only pin
    device memory.  A result with host orbitals is left as it is.
    """

    release = getattr(
        getattr(result, "wavefunctions", None), "release_intermediate_storage", None
    )
    if callable(release):
        release()


__all__ = ["IonicBackend", "periodic_backend", "reference_backend", "release_orbitals"]
