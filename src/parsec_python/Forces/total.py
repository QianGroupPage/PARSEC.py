"""Assembly of the atomic forces of a converged isolated calculation.

The order is PARSEC's: ion-ion forces (``forceion.f90``), the three local
pseudopotential terms (``forcloc.f90``), the nonlocal term
(``forcnloc.f90``), symmetrization over the point group of the atoms
(``symm_forces.f90``) and removal of the net force.  The torque that
``forcnloc.f90`` reports is taken, as there, from the symmetrized forces
before the net force is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ion_ion import ewald_ion_ion_forces, ion_ion_forces
from .local import LocalForceTerms, local_forces
from .nonlocal_force import (
    build_nonlocal_projector_gradients,
    build_periodic_projector_gradients,
    nonlocal_forces,
)
from .periodic_local import periodic_local_forces
from .symmetry import (
    AtomSymmetryOperation,
    atom_symmetry_operations,
    remove_net_force,
    symmetrize_forces,
    total_torque,
)


@dataclass(frozen=True)
class ForceResult:
    """Atomic forces in Ry/bohr with PARSEC's breakdown.

    Every array has shape ``(atoms, 3)`` unless noted.  ``total`` is what a
    relaxation or a dynamics step uses: symmetrized, with zero net force.
    ``phase_seconds`` holds, where the path that computed the forces measured
    them, the wall seconds of its parts.
    """

    positions: np.ndarray
    ion_ion: np.ndarray
    local: LocalForceTerms
    nonlocal_pseudopotential: np.ndarray
    unsymmetrized: np.ndarray
    symmetrized: np.ndarray
    net_force: np.ndarray
    torque: np.ndarray
    total: np.ndarray
    symmetry_operations: int = 1
    seconds: float = 0.0
    phase_seconds: dict | None = None

    @property
    def local_with_ion_ion(self) -> np.ndarray:
        """What PARSEC prints as "local force components"."""

        return self.ion_ion + self.local.total

    @property
    def maximum(self) -> float:
        """Largest force magnitude over the atoms."""

        return float(np.max(np.linalg.norm(self.total, axis=1), initial=0.0))


def assemble_forces(
    positions: np.ndarray,
    ion_ion: np.ndarray,
    local: LocalForceTerms,
    nonlocal_pseudopotential: np.ndarray,
    operations: Sequence[AtomSymmetryOperation] | None = None,
    *,
    seconds: float = 0.0,
    phase_seconds: dict | None = None,
    periodic: bool = False,
) -> ForceResult:
    """Combine the force terms as ``forcnloc.f90`` does at its end.

    ``periodic`` is for a crystal: the torque, which has no meaning in a periodic cell, is reported as zero.
    """

    unsymmetrized = ion_ion + local.total + nonlocal_pseudopotential
    symmetrized = (
        unsymmetrized.copy()
        if operations is None
        else symmetrize_forces(unsymmetrized, operations)
    )
    total, net = remove_net_force(symmetrized)
    return ForceResult(
        positions=np.asarray(positions, dtype=np.float64),
        ion_ion=ion_ion,
        local=local,
        nonlocal_pseudopotential=nonlocal_pseudopotential,
        unsymmetrized=unsymmetrized,
        symmetrized=symmetrized,
        net_force=net,
        torque=np.zeros(3) if periodic else total_torque(positions, symmetrized),
        total=total,
        symmetry_operations=1 if operations is None else len(operations),
        seconds=seconds,
        phase_seconds=phase_seconds,
    )


def compute_forces(
    system,
    result,
    *,
    symmetrize: bool = True,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished SCF calculation on the full grid.

    ``system`` is the prepared isolated system and ``result`` the outcome of
    its SCF loop with full-grid orbitals.  ``operations`` are the symmetry
    operations the forces are averaged over; without them the operations of
    the present atoms are found.  PARSEC finds its operations once, for the
    input structure, and a relaxation passes those at every step.
    ``symmetrize=False`` keeps the raw forces apart from the removal of the
    net force.

    A spin-polarized result needs no other formulas.  ``forcloc.f90`` reads
    the total density and the averages over the two channels of the
    potential difference and of the exchange-correlation potential, which
    are the fields of such a result; ``forcnloc.f90`` sums the nonlocal term
    over the states of both channels with the weight ``4/nspin``, which is
    that of the unpolarized case for half the occupations.
    """

    from time import perf_counter

    started = perf_counter()
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)

    repulsion = ion_ion_forces(atoms, system.pseudopotentials)
    local = local_forces(
        system.grid,
        atoms,
        system.pseudopotentials,
        specifications,
        result.density,
        np.asarray(result.output_effective_potential)
        - np.asarray(result.input_effective_potential),
        result.xc_potential,
    )
    gradients = build_nonlocal_projector_gradients(
        system.grid, atoms, system.pseudopotentials, specifications
    )
    separable = nonlocal_forces(
        system.nonlocal_operator,
        gradients,
        result.wavefunctions,
        (
            result.occupations
            if getattr(result, "spin", None) is None
            else 0.5 * np.asarray(result.occupations)
        ),
        len(atoms),
    )
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = atom_symmetry_operations(atoms)
    return assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        seconds=perf_counter() - started,
    )


def compute_periodic_forces(
    system,
    result,
    *,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished Gamma-point calculation of a periodic cell on the full grid.

    ``system`` is the prepared periodic system and ``result`` the outcome of its SCF loop with full-grid orbitals.
    The terms are those of ``forpbc.f90``, ``force_loc_pb.f90`` and ``forcnloc.f90``: the Ewald ion-ion force, the
    local ionic and core-correction forces (there is no term for the SCF residual in the periodic path) and the
    nonlocal force with the periodic projectors; then the net force is removed.  ``operations`` are optional
    symmetry operations of the atoms to average over (none by default: the Gamma-only periodic path does not
    reduce the grid by symmetry).
    """

    from time import perf_counter

    started = perf_counter()
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)

    repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
    local = periodic_local_forces(
        system.grid, atoms, system.pseudopotentials, specifications, lattice, result.density, result.xc_potential
    )
    gradients = build_periodic_projector_gradients(
        system.grid, atoms, system.pseudopotentials, specifications, lattice
    )
    separable = nonlocal_forces(
        system.nonlocal_operator,
        gradients,
        result.wavefunctions,
        result.occupations if getattr(result, "spin", None) is None else 0.5 * np.asarray(result.occupations),
        len(atoms),
    )
    return assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        seconds=perf_counter() - started,
        periodic=True,
    )


__all__ = ["ForceResult", "assemble_forces", "compute_forces", "compute_periodic_forces"]
