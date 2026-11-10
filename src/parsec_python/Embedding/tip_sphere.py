"""The FDET tip sphere of PARSEC 1.5 (``boundary_sphere_around_tip``, ``fdet_tipsphere.F90``).

With ``Boundary_Sphere_Around_Tip`` the tip run uses a small cluster sphere of
its own (the ``Boundary_Sphere_Radius`` of the tip input) instead of the
large sphere of the sample ("spot") run:

1. The tip atoms are moved by a whole number of grid steps ``S`` so that their
   (unweighted) centre lies within half a grid step of the origin,
   ``S = nint(centre/h)``; the point charges of the sample ions move with
   them.  The prepared system holds the atoms in this tip frame (the spot
   frame minus ``S h``), as PARSEC 1.5 prints them (``tipsphere_recenter``,
   lines 103-148).  A manual move gives the atoms in the spot frame again
   (``manual.dat``) and ``S`` is found anew.
2. Tip grid point ``k`` takes the sample potentials and density of spot grid
   point ``k + S``, an exact copy; points beyond the spot sphere get zero
   (``tipsphere_field``, lines 242-260).  The spot file needs the spacing of
   the tip run (621), no symmetry and no grid shift (625), and the tip grid no
   shift (622).
3. The energy (``totnrg.f90:195-196``) leaves out the sample-only constants
   (Hartree, electron-ion and kinetic energies of the sample) and
   ``E_xc[rho_A]`` on the tip grid (``tipsphere_sample_exc``), and adds the
   attraction of the tip ions by the sample electrons outside the tip sphere,
   ``E_out = h^3 sum_j rho_j sum_t (-2 Z_t / |r_j - R_t|)`` over the spot
   points ``j`` that are not tip grid points (lines 277-298, point ions:
   those points are far from the tip).  Total energies are then shifted by a
   constant of the sample: only differences are meaningful.

Slab spot files (the 2-D Ewald term, lines 329-370) are not implemented
(no reference input to check them against).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import PointCharge
from ..Output.pot_dat import PotentialFieldError, PotentialFieldFile, read_potential_field
from ..V_xc.selection import evaluate_unpolarized_xc
from .frozen_density import FrozenSampleField, check_cluster_file, kinetic_functional_of

#: Spacings that agree (``fdet_tipsphere.F90:211-213``).
SPACING_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class TipSphereState:
    """What the tip sphere adds to a :class:`FrozenSampleField`.

    ``offset`` is ``S`` (spot index = tip index + ``S``), ``missing_points``
    the tip grid points without data in the spot file, ``outside_energy``
    ``E_out`` (Ry) from ``outside_points`` spot points, ``sample_xc_energy``
    ``E_xc[rho_A]`` on the tip grid (Ry) and ``spot_point_charges`` the point
    charges of the input in the spot frame.
    """

    offset: tuple[int, int, int]
    missing_points: int
    outside_energy: float
    outside_points: int
    sample_xc_energy: float
    spot_point_charges: tuple = ()

    def report_lines(self, grid_size: int) -> list[str]:
        """The lines ``tipsphere_field`` and ``tipsphere_sample_exc`` print."""

        offset = "".join(f"{value:6d}" for value in self.offset)
        return [
            f" FDET tip sphere: spot grid offset{offset}",
            f" FDET tip sphere: tip grid points without pot.dat data{self.missing_points:9d} of{grid_size:9d}",
            f" FDET tip sphere: tip-ion/sample-elec. energy outside{self.outside_energy:20.10f} Ry "
            f"({self.outside_points:9d} spot points)",
            f" FDET tip sphere: Exc[rho_sample] on the tip grid{self.sample_xc_energy:20.10f} Ry",
        ]


def tip_offset(positions, spacing: float) -> np.ndarray:
    """``S = nint(centre/h)`` of the unweighted centre of ``positions`` (``tipsphere_recenter``, lines 114-117).

    The centre is ``sum/n`` in the order of the atoms and ``nint`` is Fortran's, halves away from zero
    (``numpy.rint`` rounds them to even: a centre at 6.5 h gives 7, not 6).
    """

    from ..Grid.space_group import fortran_nint

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    count = positions.shape[0]
    offset = []
    for axis in range(3):
        total = 0.0
        for value in positions[:, axis]:
            total += float(value)
        offset.append(fortran_nint((total / count) / float(spacing)))
    return np.array(offset, dtype=np.int64)


def to_tip_frame(atoms, point_charges, offset, spacing: float):
    """Atoms and point charges moved by ``-S h`` (``tipsphere_recenter``)."""

    from dataclasses import replace

    displacement = np.asarray(offset, dtype=np.float64) * float(spacing)
    moved_atoms = tuple(replace(atom, position=np.asarray(atom.position, dtype=np.float64) - displacement) for atom in atoms)
    moved_charges = tuple(
        PointCharge(charge.charge, np.asarray(charge.position, dtype=np.float64) - displacement)
        for charge in point_charges
    )
    return moved_atoms, moved_charges


def check_spot_file(data: PotentialFieldFile, grid, path) -> None:
    """The checks 621, 622, 624 and 625 of ``read_spot_field``/``tipsphere_field``."""

    if data.boundary_code == 3:
        raise PotentialFieldError(
            0, f"{path} is a slab pot.dat: the slab samples of the tip sphere (2-D Ewald term) are not implemented"
        )
    if data.boundary_code != 0:
        raise PotentialFieldError(624, f"FDET tip sphere needs a cluster or slab pot.dat, {path} is not one")
    check_cluster_file(data, path)
    if data.ntrans != 1 or np.any(np.abs(np.asarray(data.shift)) > 1.0e-10):
        raise PotentialFieldError(625, f"FDET tip sphere needs a pot.dat without symmetry and grid shift ({path})")
    if np.any(np.abs(np.asarray(data.steps) - float(grid.spacing)) > SPACING_TOLERANCE):
        raise PotentialFieldError(621, f"FDET tip sphere: grid spacing {grid.spacing} differs from pot.dat {path}")
    if np.any(np.abs(np.asarray(grid.settings.shift, dtype=np.float64)) > 1.0e-10):
        raise PotentialFieldError(622, "FDET tip sphere needs an unshifted grid (Ignore_Symmetry: true)")


def spot_field_on_tip_grid(data: PotentialFieldFile, grid, offset) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """``(rho_A, v_A,ion, v_H^A, missing, on_tip)`` for the offset ``S`` (lines 242-260).

    ``on_tip[j]`` says whether spot point ``j`` is a tip grid point, for the
    outside sum.
    """

    spot = np.asarray(data.points, dtype=np.int64)
    offset = np.asarray(offset, dtype=np.int64)
    rows = grid.rows_for_integer_coordinates(spot - offset)
    on_tip = rows >= 0
    density = np.zeros(grid.size, dtype=np.float64)
    ionic = np.zeros(grid.size, dtype=np.float64)
    hartree = np.zeros(grid.size, dtype=np.float64)
    density[rows[on_tip]] = data.density[on_tip]
    ionic[rows[on_tip]] = data.ionic_potential[on_tip]
    hartree[rows[on_tip]] = data.hartree_potential[on_tip]
    missing = int(grid.size - np.count_nonzero(on_tip))
    return density, ionic, hartree, missing, on_tip


def outside_energy(data: PotentialFieldFile, on_tip: np.ndarray, atoms, pseudopotentials, offset, spacing: float):
    """``E_out`` of lines 277-298 (Ry) and the number of spot points in it.

    ``atoms`` are in the tip frame; the tip ions sit at ``R_t + S h`` in the
    spot frame, the spot points at ``k_j h`` (no shift).
    """

    h = float(spacing)
    rho = np.asarray(data.density, dtype=np.float64)
    use = (~on_tip) & (rho != 0.0)
    if not np.any(use):
        return 0.0, 0
    points = np.asarray(data.points, dtype=np.float64)[use] * h
    shift = np.asarray(offset, dtype=np.float64) * h
    potential = np.zeros(points.shape[0], dtype=np.float64)
    for atom in atoms:
        charge = float(pseudopotentials[atom.symbol].ionic_charge)
        distance = np.linalg.norm(points - (np.asarray(atom.position, dtype=np.float64) + shift), axis=1)
        potential -= 2.0 * charge / distance
    return float(np.dot(potential, rho[use]) * h**3), int(np.count_nonzero(use))


def sample_xc_energy(problem, grid, density) -> float:
    """``E_xc[rho_A]`` on the tip grid, no core density (``tipsphere_sample_exc``)."""

    return float(
        evaluate_unpolarized_xc(
            problem.scf.xc_functional,
            density,
            grid,
            None,
            ionization_energy_difference=problem.scf.ionization_energy_difference,
        ).total_energy
    )


def tip_sphere_field(
    problem,
    grid,
    atoms,
    pseudopotentials,
    offset,
    *,
    spot_point_charges=(),
    source: PotentialFieldFile | None = None,
) -> FrozenSampleField:
    """The frozen field of the tip sphere for the tip-frame ``atoms`` at offset ``S``.

    ``source`` is the spot file once it is read (PARSEC keeps it after the
    first read); without it the file of the settings is read here.
    """

    settings = problem.embedding
    data = read_potential_field(settings.field_path) if source is None else source
    check_spot_file(data, grid, settings.field_path)
    density, ionic, hartree, missing, on_tip = spot_field_on_tip_grid(data, grid, offset)
    energy, count = outside_energy(data, on_tip, atoms, pseudopotentials, offset, grid.spacing)
    state = TipSphereState(
        offset=tuple(int(value) for value in offset),
        missing_points=missing,
        outside_energy=energy,
        outside_points=count,
        sample_xc_energy=sample_xc_energy(problem, grid, density),
        spot_point_charges=tuple(spot_point_charges),
    )
    return FrozenSampleField(
        density=density,
        ionic_potential=ionic,
        hartree_potential=hartree,
        potential=ionic + hartree,
        kinetic_energy=float(data.kinetic_energy),
        functional=kinetic_functional_of(settings),
        source=data,
        tip_sphere=state,
    )


def place_tip(atoms, point_charges, spacing: float):
    """``(tip-frame atoms, tip-frame charges, S, spot-frame charges)`` of spot-frame atoms and charges."""

    offset = tip_offset([atom.position for atom in atoms], spacing)
    moved_atoms, moved_charges = to_tip_frame(atoms, point_charges, offset, spacing)
    return moved_atoms, moved_charges, offset, tuple(point_charges)


def moved_tip(system, atoms):
    """The tip sphere after a move to the spot-frame ``atoms`` (``manual.dat``).

    Returns the tip-frame atoms, their point charges and the new field
    (``tipsphere_recenter(first=.false.)`` and ``tipsphere_field`` after
    ``domove``, parsec.F90:2614-2637, 2794-2800).
    """

    field = system.embedding
    state = field.tip_sphere
    spacing = system.grid.spacing
    offset = tip_offset([atom.position for atom in atoms], spacing)
    moved_atoms, moved_charges = to_tip_frame(atoms, state.spot_point_charges, offset, spacing)
    new_field = tip_sphere_field(
        system.input,
        system.grid,
        moved_atoms,
        system.pseudopotentials,
        offset,
        spot_point_charges=state.spot_point_charges,
        source=field.source,
    )
    return moved_atoms, moved_charges, new_field


__all__ = [
    "TipSphereState",
    "check_spot_file",
    "moved_tip",
    "outside_energy",
    "place_tip",
    "sample_xc_energy",
    "spot_field_on_tip_grid",
    "tip_offset",
    "tip_sphere_field",
    "to_tip_frame",
]
