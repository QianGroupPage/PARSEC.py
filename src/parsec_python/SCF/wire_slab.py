"""Preparation of a wire (1-D periodic) or slab (2-D periodic) calculation.

The counterpart of :mod:`parsec_python.SCF.pbc` for ``Boundary_Conditions:
wire`` and ``slab`` (Fortran ``pbc%per = 1, 2``).  Every SCF driver of the
periodic path (Gamma point, k-points, collinear spin, bands) runs on the
prepared system unchanged; what differs is built here:

* the grid of :mod:`parsec_python.Grid.confined` (``adjust_step`` along the
  periodic directions, the input spacing and the cylinder or slab domain
  along the confined ones) and its finite-difference operators;
* the Hartree potential (:mod:`parsec_python.Hartree.confined`): boundary
  values of ``hartset_wire``/``hartset_slab`` and the CG solve;
* the local ionic potential ``V_short - V_G`` of ``ionpot_wire``/``ionpot_slab``
  (:mod:`parsec_python.V_ion.wire_slab`), plus the charged sheets of a slab;
* projectors, atomic and core densities summed over the images along the
  periodic directions only;
* the ion-ion energy and forces of ``forceion_wire`` or ``ewald_slab``
  (plus the charged sheets).

Fortran's order is kept (``parsec.F90:806-878``): grid, local potential with
the Gaussian charges and its neutrality check, nonlocal projectors, initial
density, core density, ion-ion energy.  Atoms are used at the coordinates of
the input (no wrapping into the cell); Fortran's ``isolat`` check refuses an
atom outside the cylinder or slab.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

import numpy as np

from ..Grid.confined import ConfinedPeriodicGrid, build_confined_periodic_grid
from ..Hamiltonian.hubbard import build_hubbard_projectors
from ..Hartree.confined import (
    ConfinedHartreePlan,
    ConfinedHartreeResult,
    plan_confined_hartree,
    solve_confined_hartree,
)
from ..Laplacian import build_gradient, build_negative_laplacian
from ..Laplacian.nonorthogonal import (
    NonorthogonalLaplacianCoefficients,
    grid_laplacian_coefficients,
)
from ..models import PreparationTimings, SinglePointInput
from ..V_ion import (
    apply_fourier_filters,
    build_nonlocal_projectors,
    check_core_density,
    ionic_charge,
    load_pseudopotentials,
    normalize_density,
    superpose_atomic_density,
)
from ..V_ion.wire_slab import (
    ShortRangeLocalPotential,
    SlabEwaldTerms,
    charged_sheet_ion_energy_and_forces,
    charged_sheet_potential,
    short_range_local_potential,
    slab_ewald,
    wire_ion_ion_energy_and_forces,
)
from .pbc import PeriodicPreparedSinglePointSystem

#: Fortran behaviours that :func:`prepare_wire_slab_single_point` can reproduce
#: for validation (each is a documented defect, fixed by default).
FORTRAN_QUIRKS = frozenset({"wire_axis_points", "slab_lowest_plane", "sheet_force"})


@dataclass(frozen=True)
class WireSlabPreparedSystem(PeriodicPreparedSinglePointSystem):
    """A prepared wire or slab system (see the module docstring).

    Beyond :class:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem`:
    ``hartree_plan`` (boundary values), ``short_range`` (``V_short`` and the
    Gaussian charges), ``gaussian_potential`` (``V_G``), ``ion_ion_forces``
    (Ry/bohr, including the charged sheets), ``charged_sheet_energy`` (Ry,
    part of ``ion_ion_energy``) and, for a slab, ``ewald`` (its terms).
    """

    hartree_plan: ConfinedHartreePlan | None = None
    short_range: ShortRangeLocalPotential | None = None
    gaussian_potential: np.ndarray | None = None
    ion_ion_forces: np.ndarray | None = None
    charged_sheet_energy: float = 0.0
    charged_sheet_forces: np.ndarray | None = None
    ewald: SlabEwaldTerms | None = None
    fortran_quirks: frozenset = field(default_factory=frozenset)

    @property
    def periodicity(self) -> int:
        return int(self.grid.periodicity)

    def solve_hartree(
        self,
        density: np.ndarray,
        initial_potential: np.ndarray | None = None,
        *,
        raise_on_nonconvergence: bool = True,
    ) -> ConfinedHartreeResult:
        return solve_confined_hartree(
            density,
            self.hartree_plan,
            self.negative_laplacian,
            self.input.hartree,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )


def check_atoms_inside(atoms, grid: ConfinedPeriodicGrid) -> None:
    """Fortran's ``isolat`` for a wire or slab (``isolat.f90:72-95``).

    An atom whose distance from the axis (wire) or the plane ``z = 0``
    (slab) exceeds ``Boundary_Sphere_Radius`` is refused (error 291).
    Deviation: Fortran compares with ``int(rmax)`` (``rmax`` is declared an
    integer there), which refuses atoms between ``int(R)`` and ``R`` too;
    here the radius itself is the limit.
    """

    radius = grid.confinement_radius
    for index, atom in enumerate(atoms, start=1):
        position = np.asarray(atom.position, dtype=np.float64)
        confined = position.copy()
        confined[: grid.periodicity] = 0.0
        if float(np.sqrt(confined @ confined)) > radius:
            raise ValueError(
                f"atom #{index} ({atom.symbol}) is outside the boundary "
                f"{'cylinder' if grid.periodicity == 1 else 'slab'} of radius "
                f"{radius:g} bohr (PARSEC isolat.f90, error 291)"
            )


def prepare_wire_slab_single_point(
    problem: SinglePointInput,
    *,
    fortran_quirks: frozenset[str] | set[str] = frozenset(),
) -> WireSlabPreparedSystem:
    """Build every static component of a wire or slab calculation.

    ``problem.grid`` must be :class:`~parsec_python.models.PeriodicGridSettings`
    with ``periodicity`` 1 or 2.  ``fortran_quirks`` (a subset of
    :data:`FORTRAN_QUIRKS`) reproduces Fortran defects for validation:
    ``"wire_axis_points"`` and ``"slab_lowest_plane"`` (Hartree boundary
    values), ``"sheet_force"`` (the charged-sheet force without ``2 pi``).
    """

    unknown = set(fortran_quirks) - FORTRAN_QUIRKS
    if unknown:
        raise ValueError(f"unknown Fortran quirks: {sorted(unknown)}")
    quirks = frozenset(fortran_quirks)
    if problem.periodic_cell is None or getattr(problem.grid, "periodicity", 3) not in (1, 2):
        raise ValueError("prepare_wire_slab_single_point needs a wire or slab input")
    preparation_start = time.perf_counter()
    atoms = tuple(problem.atoms)
    stage_start = time.perf_counter()
    pseudopotentials = load_pseudopotentials(
        problem.pseudopotentials, xc_functional=problem.scf.xc_functional
    )
    pseudopotential_loading_seconds = time.perf_counter() - stage_start
    electron_count = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    if electron_count <= 0:
        raise ValueError("the requested system has no valence electrons")

    stage_start = time.perf_counter()
    grid = build_confined_periodic_grid(problem.periodic_cell, problem.grid)
    check_atoms_inside(atoms, grid)
    grid_seconds = time.perf_counter() - stage_start
    # fourier_f.f90 filters with grid%step(1) for every periodic system.
    pseudopotentials = apply_fourier_filters(
        pseudopotentials, problem.pseudopotentials, float(grid.spacings[0])
    )
    laplacian_coefficients: NonorthogonalLaplacianCoefficients = grid_laplacian_coefficients(grid)
    # Symmetrize_Charge_Density: the space group of the wire or slab as
    # permutations of its grid points (symm_scalar of newrho.F90).
    density_symmetry = None
    if problem.scf.symmetrize_density:
        from ..Grid.space_group import confined_grid_symmetry, identity_space_group

        group = getattr(problem, "space_group", None) or identity_space_group()
        density_symmetry = confined_grid_symmetry(grid, group)

    stage_start = time.perf_counter()
    negative_laplacian = build_negative_laplacian(grid)
    gradient = build_gradient(grid)
    hartree_plan = plan_confined_hartree(
        grid, problem.hartree.multipole_order, fortran_quirks=quirks
    )
    finite_difference_seconds = time.perf_counter() - stage_start

    parts = _atom_dependent_parts(
        problem, grid, pseudopotentials, negative_laplacian, hartree_plan, atoms, quirks
    )
    timings = PreparationTimings(
        pseudopotential_loading_seconds=pseudopotential_loading_seconds,
        grid_seconds=grid_seconds,
        finite_difference_seconds=finite_difference_seconds,
        local_ionic_seconds=parts.pop("local_ionic_seconds"),
        nonlocal_ionic_seconds=parts.pop("nonlocal_ionic_seconds"),
        initial_density_seconds=parts.pop("initial_density_seconds"),
        core_density_seconds=parts.pop("core_density_seconds"),
        ion_ion_seconds=parts.pop("ion_ion_seconds"),
        total_seconds=time.perf_counter() - preparation_start,
    )
    return WireSlabPreparedSystem(
        input=problem,
        atoms=atoms,
        electron_count=electron_count,
        pseudopotentials=pseudopotentials,
        grid=grid,
        negative_laplacian=negative_laplacian,
        gradient=gradient,
        alpha_z_energy=0.0,
        timings=timings,
        laplacian_coefficients=laplacian_coefficients,
        hartree_plan=hartree_plan,
        fortran_quirks=quirks,
        density_symmetry=density_symmetry,
        **parts,
    )


def _atom_dependent_parts(
    problem: SinglePointInput,
    grid: ConfinedPeriodicGrid,
    pseudopotentials,
    negative_laplacian,
    hartree_plan: ConfinedHartreePlan,
    atoms: tuple,
    quirks: frozenset,
) -> dict:
    """What a wire or slab rebuilds for its atoms, in Fortran's order.

    ``parsec.F90:806-878`` at the setup and l.1930-2046 after every move:
    the local potential ``V_short - V_G`` (``ionpot_wire``/``ionpot_slab``,
    plus the fixed charged sheets), the projectors (``nonloc``), the
    superposed atomic density, the core density (``corecd``) and the ion-ion
    energy and forces (``forceion_wire``/``ewald_slab`` plus the sheets).
    :func:`prepare_wire_slab_single_point` and
    :func:`update_wire_slab_geometry` share it, so a moved structure is
    prepared exactly as a fresh one.  Returns the fields of
    :class:`WireSlabPreparedSystem` and the stage timings.
    """

    # Local potential: V_short - V_G (ionpot_wire.f90 / ionpot_slab.f90).
    stage_start = time.perf_counter()
    short_range = short_range_local_potential(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    # hpotcg starts from the Gaussian density itself (ionpot_wire.f90:223).
    gaussian_solution = solve_confined_hartree(
        short_range.gaussian_density,
        hartree_plan,
        negative_laplacian,
        problem.hartree,
        short_range.gaussian_density,
    )
    gaussian_potential = gaussian_solution.potential
    ionic_potential = short_range.potential - gaussian_potential
    if problem.charged_sheets:
        ionic_potential = ionic_potential + charged_sheet_potential(grid, problem.charged_sheets)
    local_ionic_seconds = time.perf_counter() - stage_start

    lattice_vectors = problem.periodic_cell.lattice_vectors
    stage_start = time.perf_counter()
    nonlocal_operator = build_nonlocal_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
    )
    # LDA+U projectors (upot.F90): images along the periodic vectors only
    # (nreplica(1:pbc%per) = 1); None without LDA+U.
    hubbard = build_hubbard_projectors(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        np.asarray(lattice_vectors)[: grid.periodicity],
    )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if problem.initial_density_settings.method != "sad":
        raise NotImplementedError(
            "a wire or slab only supports initial_density_settings.method='sad'"
        )
    electron_count = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    initial_density = superpose_atomic_density(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
    )
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, electron_count)
    initial_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    core_density = check_core_density(
        superpose_atomic_density(
            grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, core=True
        )
    )
    core_density_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    ewald = None
    if grid.periodicity == 1:
        repulsion, ion_forces = wire_ion_ion_energy_and_forces(
            atoms, pseudopotentials, grid.period
        )
    else:
        ewald = slab_ewald(atoms, pseudopotentials, lattice_vectors)
        repulsion, ion_forces = ewald.energy, ewald.forces
    sheet_energy = 0.0
    sheet_forces = None
    if problem.charged_sheets:
        sheet_energy, sheet_forces = charged_sheet_ion_energy_and_forces(
            atoms,
            pseudopotentials,
            problem.charged_sheets,
            grid.surface_area,
            fortran_force="sheet_force" in quirks,
        )
        repulsion = repulsion + sheet_energy
        ion_forces = ion_forces + sheet_forces
    atomic_reference_correction = float(
        sum(problem.pseudopotentials[atom.symbol].atomic_energy_correction for atom in atoms)
    )
    ion_ion_seconds = time.perf_counter() - stage_start
    return {
        "ionic_potential": ionic_potential,
        "nonlocal_operator": nonlocal_operator,
        "hubbard": hubbard,
        "initial_density": initial_density,
        "core_density": core_density,
        "ion_ion_energy": float(repulsion),
        "atomic_reference_correction": atomic_reference_correction,
        "short_range": short_range,
        "gaussian_potential": gaussian_potential,
        "ion_ion_forces": ion_forces,
        "charged_sheet_energy": float(sheet_energy),
        "charged_sheet_forces": sheet_forces,
        "ewald": ewald,
        "local_ionic_seconds": local_ionic_seconds,
        "nonlocal_ionic_seconds": nonlocal_ionic_seconds,
        "initial_density_seconds": initial_density_seconds,
        "core_density_seconds": core_density_seconds,
        "ion_ion_seconds": ion_ion_seconds,
    }


def wire_slab_back_to_cell(atoms, lattice_vectors, periodicity: int) -> tuple:
    """``back_to_cell`` of ``initial.f90:651-696`` for a wire or slab.

    Along every periodic direction the fractional coordinate ``u`` becomes
    ``u - nint(u)`` (Fortran's cell is centred on the origin, ``[-1/2,
    1/2]``); the confined Cartesian components are kept.  PARSEC calls it
    after every ``domove`` and ``moldyn`` of a periodic system.  An atom
    that needs no translation keeps its object and coordinates; one that
    does is moved by whole lattice vectors (``r - sum_i nint(u_i) a_i``;
    Fortran recomputes ``r`` as ``latt_vec u``, which differs in the last
    bits only).
    """

    from ..Grid.space_group import fortran_nint

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    placed = []
    for atom in atoms:
        position = np.asarray(atom.position, dtype=np.float64)
        fractional = position @ inverse
        steps = [fortran_nint(float(fractional[axis])) for axis in range(periodicity)]
        if not any(steps):
            placed.append(atom)
            continue
        moved = position.copy()
        for axis, step in enumerate(steps):
            if step:
                moved = moved - float(step) * lattice[axis]
        placed.append(replace(atom, position=moved))
    return tuple(placed)


def update_wire_slab_geometry(system: WireSlabPreparedSystem, atoms) -> WireSlabPreparedSystem:
    """Return ``system`` with its atoms at new positions on the same grid.

    What ``parsec.F90`` (l.1930-2046) repeats for a wire or slab after
    ``domove``/``moldyn``: ``back_to_cell`` (:func:`wire_slab_back_to_cell`),
    ``isolat`` (an atom outside the cylinder or slab stops the run, error
    291), ``nonloc``, ``corecd``, ``ionpot_wire``/``ionpot_slab`` and
    ``forceion_wire``/``ewald_slab``, through the same function as
    :func:`prepare_wire_slab_single_point`: the result equals a fresh
    preparation of the translated atoms bit for bit.  Grid, Laplacian,
    gradient, Hartree plan, pseudopotentials, electron count and the
    density symmetrization are shared.  ``system.input.atoms`` holds the
    positions as given (the optimizer and the integrator keep them
    unwrapped), ``atoms`` of the result the translated ones; the starting
    spin densities are rebuilt where ``system`` has them.
    """

    started = time.perf_counter()
    atoms = tuple(atoms)
    if len(atoms) != len(system.atoms) or any(
        new.symbol != old.symbol for new, old in zip(atoms, system.atoms)
    ):
        raise ValueError("moved atoms must keep their number, order and species")
    problem = replace(system.input, atoms=atoms)
    grid = system.grid
    placed = wire_slab_back_to_cell(
        atoms, problem.periodic_cell.lattice_vectors, grid.periodicity
    )
    check_atoms_inside(placed, grid)
    parts = _atom_dependent_parts(
        problem,
        grid,
        system.pseudopotentials,
        system.negative_laplacian,
        system.hartree_plan,
        placed,
        system.fortran_quirks,
    )
    timings = PreparationTimings(
        local_ionic_seconds=parts.pop("local_ionic_seconds"),
        nonlocal_ionic_seconds=parts.pop("nonlocal_ionic_seconds"),
        initial_density_seconds=parts.pop("initial_density_seconds"),
        core_density_seconds=parts.pop("core_density_seconds"),
        ion_ion_seconds=parts.pop("ion_ion_seconds"),
    )
    moved = replace(system, input=problem, atoms=placed, initial_spin_densities=None, **parts)
    if system.initial_spin_densities is not None:
        from .pbc import periodic_spin_start

        moved = replace(moved, initial_spin_densities=periodic_spin_start(moved))
    return replace(
        moved,
        timings=replace(timings, total_seconds=time.perf_counter() - started),
    )


def is_wire_or_slab(problem) -> bool:
    """Whether a :class:`~parsec_python.models.SinglePointInput` is a wire or slab."""

    return problem.periodic_cell is not None and getattr(problem.grid, "periodicity", 3) in (1, 2)


__all__ = [
    "FORTRAN_QUIRKS",
    "WireSlabPreparedSystem",
    "check_atoms_inside",
    "is_wire_or_slab",
    "prepare_wire_slab_single_point",
    "update_wire_slab_geometry",
    "wire_slab_back_to_cell",
]
