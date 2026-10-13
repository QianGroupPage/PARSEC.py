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
    the form ``run_scf`` accepts.
    """

    run_scf: Callable[..., object]
    compute_forces: Callable[..., object]
    update_geometry: Callable[[object, Sequence[Atom]], object]
    warm_start: Callable[[object, object, object], object]


def reference_backend() -> IonicBackend:
    """The readable full-grid NumPy/SciPy path."""

    from ..Forces import compute_forces
    from ..SCF import run_scf, update_geometry, warm_start_for_moved_atoms

    return IonicBackend(
        run_scf=run_scf,
        compute_forces=compute_forces,
        update_geometry=update_geometry,
        warm_start=warm_start_for_moved_atoms,
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


__all__ = ["IonicBackend", "reference_backend", "release_orbitals"]
