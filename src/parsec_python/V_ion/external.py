"""External potentials of a confined system: point charges and a uniform field.

Both enter the Hamiltonian as part of the local ionic potential ``pot%vion``
of PARSEC, so they appear in the electron-ion energy and in every effective
potential, and nowhere else in the electronic problem.  Lengths are in bohr,
charges in e, potentials and energies in Rydberg.

Point charges (``Add_Point_Charges``; ``ionpot.f90`` lines 134-175 and
``forcept.f90``)
    A classical charge ``q`` at ``R_q`` adds the electron potential

    ``V_pc(r) = -2 q / |r - R_q|``   for ``|r - R_q| > 1 bohr``, else ``0``,

    (the factor two converts Hartree to Rydberg; PARSEC leaves the grid
    points within one bohr of the charge without it to avoid the singularity)
    and the energy of its interaction with the pseudo-ions,

    ``E_QN = sum_q sum_a 2 q Z_a / |R_a - R_q|``,

    which PARSEC adds to the ion-ion energy ``enuc``.  The interaction of
    the point charges with each other is commented out in ``forcept.f90``
    and is not part of the energy here either.  The force of the point
    charges on the atoms is in :func:`parsec_python.Forces.ion_ion.point_charge_forces`.

Uniform electric field (``Polarizability``; ``polar.f90``)
    For a field of strength ``F`` along the unit direction ``d`` PARSEC adds

    ``V_field(r) = 2 F d . r``

    with ``r = (shift + k) h`` measured from the origin of the grid.  The
    factor two converts Hartree to Rydberg: the number ``F`` is used as a
    field in Hartree atomic units (e/bohr^2, 5.142e11 V/m), although the
    input labels it ``ry/bohr/e`` and ``parsec.out`` prints ``[Ry/au]``.
    The polarizabilities ``polar.f90`` computes from the energies in Hartree
    and the dipoles in e bohr are consistent with that reading.  This module
    keeps the Fortran convention.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..models import Atom, PointCharge

#: Grid points closer than this to a point charge (bohr) get no potential
#: from it (``ionpot.f90``: ``if (dist > one)``).
POINT_CHARGE_CORE_RADIUS = 1.0


def _charges_and_positions(point_charges: Sequence[PointCharge]):
    charges = np.array([item.charge for item in point_charges], dtype=np.float64)
    positions = np.array(
        [item.position for item in point_charges], dtype=np.float64
    ).reshape(-1, 3)
    return charges, positions


def point_charge_potential(
    coordinates: np.ndarray,
    point_charges: Sequence[PointCharge],
) -> np.ndarray:
    """``sum_q V_pc`` on the grid points ``coordinates`` ``(points, 3)``, in Ry.

    Each charge contributes ``-2 q/|r - R_q|`` where the distance exceeds
    :data:`POINT_CHARGE_CORE_RADIUS` and nothing elsewhere.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    total = np.zeros(coordinates.shape[0], dtype=np.float64)
    for item in point_charges:
        distance = np.linalg.norm(coordinates - np.asarray(item.position), axis=1)
        outside = distance > POINT_CHARGE_CORE_RADIUS
        total[outside] += -2.0 * float(item.charge) / distance[outside]
    return total


def point_charge_ion_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, object],
    point_charges: Sequence[PointCharge],
) -> float:
    """``E_QN = sum_q sum_a 2 q Z_a/|R_a - R_q|`` in Ry (``forcept.f90``).

    ``potentials`` maps each atom symbol to an object with ``ionic_charge``.
    """

    if not point_charges:
        return 0.0
    charges, positions = _charges_and_positions(point_charges)
    energy = 0.0
    for atom in atoms:
        ionic = float(potentials[atom.symbol].ionic_charge)
        distance = np.linalg.norm(np.asarray(atom.position) - positions, axis=1)
        if np.any(distance <= 0.0):
            raise ValueError("a point charge lies on a nucleus")
        energy += float(np.sum(2.0 * charges * ionic / distance))
    return energy


def shifted_point_charges(
    point_charges: Sequence[PointCharge], shift: np.ndarray | Sequence[float]
) -> tuple[PointCharge, ...]:
    """The point charges translated by ``-shift`` (``initial.f90`` 184-196).

    PARSEC moves the point charges with the atoms when it centres the
    bounding box of the cluster at the origin.
    """

    shift = np.asarray(shift, dtype=np.float64)
    return tuple(
        PointCharge(item.charge, np.asarray(item.position) - shift)
        for item in point_charges
    )


def electric_field_potential(
    coordinates: np.ndarray,
    field: float,
    direction: Sequence[float],
) -> np.ndarray:
    """``V_field = 2 F d . r`` in Ry on ``coordinates`` ``(points, 3)``.

    ``field`` is ``F`` as PARSEC uses it (Hartree atomic units, see the
    module docstring) and ``direction`` the vector ``d``; ``polar.f90``
    takes one of the six unit vectors along the Cartesian axes.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    return 2.0 * float(field) * (coordinates @ direction)


__all__ = [
    "POINT_CHARGE_CORE_RADIUS",
    "electric_field_potential",
    "point_charge_ion_energy",
    "point_charge_potential",
    "shifted_point_charges",
]
