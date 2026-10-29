"""Atomic forces of a calculation that ran on an accelerated backend.

The assembly is :func:`parsec_python.Forces.assemble_forces`; what differs
from the readable path is where the two expensive sums are formed.  The
local-pseudopotential terms couple every atom with every grid point and run
on a GPU when the calculation did.  The nonlocal term needs every occupied
orbital: it is contracted on the devices and MPI ranks that hold the
orbitals, in their symmetry sectors, and only one small array per rank comes
back.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Sequence

import numpy as np

from parsec_python.Forces import (
    AtomSymmetryOperation,
    ForceResult,
    assemble_forces,
    atom_symmetry_operations,
    ewald_ion_ion_forces,
    ion_ion_forces,
    local_forces,
    nonlocal_forces,
)


def sector_columns(representations: np.ndarray) -> np.ndarray:
    """Column of every selected state inside its own symmetry sector.

    The states are listed in global eigenvalue order and each sector keeps
    its own in ascending order, so a state is the ``k``-th of its sector when
    ``k`` states of that sector come before it.
    """

    representations = np.asarray(representations)
    columns = np.zeros(representations.size, dtype=np.int32)
    for representation in np.unique(representations):
        members = np.flatnonzero(representations == representation)
        columns[members] = np.arange(members.size, dtype=np.int32)
    return columns


def _atom_sums(terms: np.ndarray, labels, atom_count: int) -> np.ndarray:
    owners = np.fromiter((label[0] for label in labels), dtype=np.int64, count=len(labels))
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    for component in range(3):
        forces[:, component] = np.bincount(
            owners, weights=terms[:, component], minlength=atom_count
        )
    return forces


def _nonlocal_forces(system, result, clock: dict) -> np.ndarray:
    atom_count = len(system.atoms)
    operator = system.nonlocal_operator
    if operator.projectors.shape[1] == 0:
        return np.zeros((atom_count, 3), dtype=np.float64)
    occupations = np.asarray(result.occupations, dtype=np.float64)
    channels = getattr(result, "spin", None)
    if channels is not None:
        return _spin_nonlocal_forces(system, channels, operator, atom_count)

    sectors = getattr(system.backend, "symmetry_eigensolver", None)
    if sectors is not None:
        # Sector labels of the reference result are PARSEC's, from one.
        representations = np.asarray(result.representations, dtype=np.int32) - 1
        terms = sectors.projector_force_terms(
            representations, sector_columns(representations), occupations
        )
        # What this process spent inside its own part of the command.
        for name, seconds in (getattr(sectors, "force_phase_seconds", None) or {}).items():
            clock[f"nonlocal_{name}"] = seconds
        return _atom_sums(terms, operator.labels, atom_count)

    gradients = system.projector_gradients()
    device_solver = getattr(system.eigenproblem_solver, "solver", None)
    device_operator = getattr(system.backend, "device_operator", None)
    state = None if device_solver is None else device_solver.device_state
    if state is not None and device_operator is not None:
        from .nonlocal_device import gradient_transposes, projector_force_terms

        occupied = np.flatnonzero(occupations != 0.0)
        if not occupied.size:
            return np.zeros((atom_count, 3), dtype=np.float64)
        weights = occupations[: int(occupied[-1]) + 1]
        terms = projector_force_terms(
            device_operator,
            state.subspace.vectors,
            weights,
            gradient_transposes(gradients.components),
        )
        return _atom_sums(terms, operator.labels, atom_count)
    return nonlocal_forces(
        operator, gradients, result.wavefunctions, occupations, atom_count
    )


def _spin_nonlocal_forces(system, channels, operator, atom_count: int) -> np.ndarray:
    """Nonlocal forces of the two channels of a spin-polarized result.

    ``forcnloc.f90`` weights a state by four over the number of spin
    channels: each channel enters the unpolarized contraction with half its
    occupations.  The orbitals are read where the channel keeps them, on
    its device or in the result.
    """

    gradients = system.projector_gradients()
    for_channel = getattr(system.eigenproblem_solver, "for_spin_channel", None)
    device_operator = getattr(system.backend, "device_operator", None)
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    transposes = None
    for index, channel in enumerate(channels):
        occupations = 0.5 * np.asarray(channel.occupations, dtype=np.float64)
        occupied = np.flatnonzero(occupations != 0.0)
        if not occupied.size:
            continue
        device_solver = (
            getattr(for_channel(index), "solver", None)
            if callable(for_channel)
            else None
        )
        state = None if device_solver is None else device_solver.device_state
        if state is not None and device_operator is not None:
            from .nonlocal_device import gradient_transposes, projector_force_terms

            if transposes is None:
                transposes = gradient_transposes(gradients.components)
            terms = projector_force_terms(
                device_operator,
                state.subspace.vectors,
                occupations[: int(occupied[-1]) + 1],
                transposes,
            )
            forces += _atom_sums(terms, operator.labels, atom_count)
        else:
            forces += nonlocal_forces(
                operator, gradients, channel.wavefunctions, occupations, atom_count
            )
    return forces


def _array_module(system):
    """``(xp, erf, device)`` of the sums of a periodic cell: CuPy on the current device when the calculation ran on a GPU."""

    if system.backend_info.selected != "cupy":
        return np, None, None
    from cupyx.scipy.special import erf

    from ..backends.cupy import require_cupy
    from .local_device import local_force_device_ids

    cp, _ = require_cupy()
    return cp, erf, local_force_device_ids(system)[0]


def compute_periodic_forces(
    system,
    result,
    *,
    symmetrize: bool = False,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished accelerated SCF calculation of a periodic cell at the Gamma point.

    The terms are those of :func:`parsec_python.Forces.compute_periodic_forces`.  The Ewald ion-ion force is formed on
    the host.  The local ionic and core-correction forces are the ball sums and the one FFT of
    :func:`.periodic_local.fast_periodic_local_forces`, on the GPU when the calculation ran on one.  The nonlocal term is
    contracted where the orbitals are, as for an isolated structure, with the gradients of the periodic projectors.
    A crystal has no symmetrization here, no torque, and its net force is removed.
    """

    if symmetrize or operations:
        raise ValueError(
            "a periodic cell has no symmetrized forces: its structure is not reduced by space-group operations"
        )
    from .periodic_local import fast_periodic_local_forces

    started = perf_counter()
    clock: dict[str, float] = {}
    reference = system.reference
    atoms = reference.cell_atoms
    positions = np.array([atom.position for atom in reference.atoms], dtype=np.float64)
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
    clock["ion_ion"] = perf_counter() - started
    mark = perf_counter()
    xp, erf, device = _array_module(system)
    arguments = (
        system.grid,
        atoms,
        system.pseudopotentials,
        system.input.pseudopotentials,
        lattice,
        result.density,
        result.xc_potential,
    )
    if device is None:
        local = fast_periodic_local_forces(*arguments)
    else:
        with xp.cuda.Device(device):
            local = fast_periodic_local_forces(*arguments, xp=xp, erf=erf)
    clock["local"] = perf_counter() - mark
    mark = perf_counter()
    separable = _nonlocal_forces(system, result, clock)
    clock["nonlocal"] = perf_counter() - mark
    forces = assemble_forces(
        positions, repulsion, local, separable, None, phase_seconds=clock, periodic=True
    )
    return replace(forces, seconds=perf_counter() - started)


def compute_forces(
    system,
    result,
    *,
    symmetrize: bool = True,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished accelerated SCF calculation.

    ``system`` is the prepared accelerated system and ``result`` what its SCF
    loop returned; call this before the next eigensolve of the system, whose
    eigensolver still holds the orbitals.  The other arguments are those of
    :func:`parsec_python.Forces.compute_forces`.  In an MPI run this is called
    on the root rank.
    """

    if getattr(system.input, "periodic_cell", None) is not None:
        return compute_periodic_forces(system, result, symmetrize=symmetrize, operations=operations)
    started = perf_counter()
    clock: dict[str, float] = {}
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    repulsion = ion_ion_forces(atoms, system.pseudopotentials)
    clock["ion_ion"] = perf_counter() - started
    mark = perf_counter()
    residual = np.asarray(result.output_effective_potential) - np.asarray(
        result.input_effective_potential
    )
    fields = (
        system.grid,
        atoms,
        system.pseudopotentials,
        specifications,
        result.density,
        residual,
        result.xc_potential,
    )
    if system.backend_info.selected == "cupy":
        from .local_device import (
            cupy_local_forces,
            cupy_wedge_local_forces,
            local_force_device_ids,
            local_force_sums_setting,
        )

        devices = local_force_device_ids(system)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
        if (
            reduction is not None
            and reduction.group_order > 1
            and local_force_sums_setting() == "wedge"
        ):
            local = cupy_wedge_local_forces(*fields, reduction, device_ids=devices)
            clock["local_points"] = float(reduction.wedge_size)
        else:
            local = cupy_local_forces(*fields, device_ids=devices)
            clock["local_points"] = float(system.grid.size)
    else:
        local = local_forces(*fields)
    clock["local"] = perf_counter() - mark
    mark = perf_counter()
    separable = _nonlocal_forces(system, result, clock)
    clock["nonlocal"] = perf_counter() - mark
    mark = perf_counter()
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = atom_symmetry_operations(atoms)
    forces = assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        phase_seconds=clock,
    )
    clock["symmetrization"] = perf_counter() - mark
    return replace(forces, seconds=perf_counter() - started)


__all__ = ["compute_forces", "compute_periodic_forces", "sector_columns"]
