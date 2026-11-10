"""Spin-polarized SCF loops in the symmetry sectors (``Spin_Symmetry_Sectors``).

A spin-polarized CuPy run of the accelerated driver can solve both channels in
the representation sectors of the point group of the atoms, as an unpolarized
run does (:class:`~parsec_python.acceleration.Eigensolvers.symmetry.CuPySymmetrySCFEigensolver`,
one channel view per spin).  Every sector orbital belongs to a real
one-dimensional representation, so ``|psi|^2`` is invariant and so are both
channel densities, the total density and every potential: the loop keeps one
value per symmetry orbit for all of them (:class:`SpinSectorReducer`), solves
Hartree on the wedge and evaluates the spin functional on the orbits.

The sectors cannot represent a density that breaks the point group of the
sector representations, and they constrain a state that breaks the larger
point group of the atoms to the directions that group allows.  The full-grid
loop of a partly filled degenerate level (the oxygen atom: one minority p
electron in three degenerate p levels; an open d shell, the Sc atom with one d
electron in five) breaks the symmetry and finds a state that no sector state
holds: at 80 K the sectors keep the symmetric ensemble (one third in each p
orbital, 68 mRy above for the oxygen atom), at integer occupations an
axis-aligned partner (14 mRy above for the O atom of the fixture grid, 14.8 mRy
for the Sc atom).

The sectors can only be wrong where degenerate partners lie in different
sectors: partners within one sector mix there as freely as on the full grid.
A multiplet (:func:`sector_multiplets`) is therefore a run of levels of one
channel whose neighbours lie within :data:`START_MULTIPLET_WIDTH` = 0.01 Ry,
whose whole spread is at most :data:`START_MULTIPLET_SPREAD` = 0.03 Ry and
whose levels lie in at least two sectors.  The width is above the splitting of
physical multiplets by the grid where they are resolved (measured, PBE TM
potentials, h 0.30 bohr: the 3d levels of Sc to Fe split into e_g and t_2g by
2.6-18 mRy, each part exactly degenerate; p levels exact; 0.3 mRy at h 0.20),
and the spread bound keeps out a band of levels: the valence band of a large
cluster and the unconverged top of a first subspace, where the levels of the
first filter lie closer than the width over several eV (the nanodiamonds
C459H204 and C795H300: chains of 159 to 1818 levels through the Fermi level of
the first iteration).  The sector result is kept only where three tests pass
(each raises :class:`SpinSectorsAbandoned`, after which the driver prepares the
run again on the full grid and starts it there):

1. The start Hamiltonian (the first SCF iteration of a run, from the superposed
   atomic densities, which carries the full point group of the atoms on the
   grid): no multiplet of a channel is partly filled
   (:func:`partly_filled_multiplets`): ``n`` electrons in ``d`` levels with
   ``min(n, d - n) > 0.05`` (:data:`SPIN_SECTOR_OCCUPATION_TOLERANCE`: a
   partly filled shell holds a third of an electron or more per level, while
   the Fermi-Dirac tail of a chain of empty levels holds a few 1e-3: nine
   empty levels of C459H204+ at 500 K hold 3.4e-3 electrons).  A partly filled multiplet of fixed occupations
   (``occup.in``) that are equal on all its levels is kept: its density is
   invariant on the full grid as well.  A report counts once the levels are
   converged: the spin loop solves the start Hamiltonian again from its
   subspaces (up to :data:`START_SPECTRUM_PASSES` = 4 times) and abandons the
   sectors when the report stays and the levels up to the Fermi level moved
   by at most :data:`START_LEVEL_CHANGE` = 1e-4 Ry in the last solve (a small
   system: one further solve, its partners agree to 1e-8 Ry).  A report on
   the unconverged levels of a first filter (a large cluster) disappears or
   keeps moving, and the run goes on: tests 2 and 3 decide at the end.
2. At the end of every SCF loop without an exchange operator and of the whole
   run: each channel density is invariant under the operations of the point
   group of the atoms on the grid (the signed permutations of the axes that map
   the atoms and the grid onto themselves, also those outside the sector
   group; :func:`atom_group_images`), within
   :data:`DENSITY_SYMMETRY_TOLERANCE` = 0.01 electrons of
   ``int |rho(g r) - rho(r)| dr`` (:func:`density_symmetry_breaking`).  A sector
   state with a partly filled multiplet whose partners an operation of the
   atoms relates (the p or d levels of an atom, the pi levels of a molecule
   along an axis) breaks it by about an electron (V 1.25, Co 1.17, the O atom
   0.5 to 1); an invariant state differs by what the SCF leaves, since the
   sectors enforce only the abelian subgroup: 6e-7 to 2e-5 electrons for small
   systems, 3e-3 to 4e-3 for C459H204 and C185H124 at Convergence_Criterion
   2e-4 Ry, 1.1e-2 to 2.2e-2 at 2e-3 Ry (the looser criterion of the first
   steps of a relaxation with Relaxation_SCF_Upscale).  The test therefore
   tells the two apart by convergence: where a converged reduced loop exceeds
   the tolerance by less than :data:`DENSITY_SYMMETRY_NOISE_BOUND` = 0.2
   electrons, the loop goes on to a criterion ten times tighter (at most
   :data:`DENSITY_SYMMETRY_TIGHTENINGS` = 3 times) and measures again; noise
   shrinks with the residual, a broken state keeps its asymmetry (the loop
   stops tightening once a measure is more than half the one before), and only
   what is left above the tolerance abandons the sectors.  This also catches a
   filling the start did not show (a 4s-3d transfer during the SCF).
3. With Fermi-Dirac occupations, no level of a multiplet (above, or an exact
   one: levels within :data:`MULTIPLET_WIDTH` of each other in two or more
   sectors) holds a fractional occupation at the end, ``min(f, 1 - f) < 0.05``
   (:func:`fractional_occupations`; the symmetric ensemble of a partly filled
   shell holds a third or a half per level).  A fractional single level, or a multiplet within one sector, is
   what the full grid has too.  Fixed occupations are not tested here (test 1
   covers their multiplets).

What the tests do not see: a partly filled multiplet whose partners no grid
operation of the atoms relates (the e levels of a C3 or C6 axis) and that is
not partly filled at the start, a multiplet that only the first filter of a
large system leaves unresolved and whose sector state ends invariant and
integer, a state that passes through a partly filled multiplet during the SCF
and returns to an invariant integer one, and a symmetric state with integer
occupations that a rotation across a gap would lower (a closed shell in the
sectors has the same exposure).

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from parsec_python.Energy.total_energy import _breakdown
from parsec_python.Mixer import make_mixer
from parsec_python.Mixer.anderson import EMPTY_CHANNEL_ELECTRONS, SpinResidualMetrics
from parsec_python.V_xc.ca_lda_spin import SpinXCResult

from .symmetry_fields import SymmetryAndersonMixer, SymmetrySCFReducer, SymmetryWeightedMixer

#: Electrons below which a multiplet counts as filled or empty (tests 1 and 3): above Fermi-Dirac tails.
SPIN_SECTOR_OCCUPATION_TOLERANCE = 5.0e-2
#: Levels of a channel closer than this (Ry) are listed as one multiplet in a report.
MULTIPLET_WIDTH = 1.0e-4
#: Levels of one channel whose neighbours lie closer than this (Ry) form one group (tests 1 and 3, see the module).
START_MULTIPLET_WIDTH = 1.0e-2
#: The largest spread (Ry) of such a group that counts as a multiplet: a longer chain is a band of levels.
START_MULTIPLET_SPREAD = 3.0e-2
#: Further solves of the start Hamiltonian before a partly filled multiplet of test 1 counts (converged levels).
START_SPECTRUM_PASSES = 4
#: Largest change (Ry) of the levels up to the Fermi level between two solves of the start Hamiltonian.
START_LEVEL_CHANGE = 1.0e-4
#: Electrons by which a channel density may differ from its image under an operation of the atoms.
DENSITY_SYMMETRY_TOLERANCE = 1.0e-2
#: Above this many electrons of asymmetry a state is broken, whatever the SCF residual (test 2).
DENSITY_SYMMETRY_NOISE_BOUND = 0.2
#: Times a converged reduced loop may tighten its criterion tenfold to see whether an asymmetry is noise.
DENSITY_SYMMETRY_TIGHTENINGS = 3
_RYDBERG_TO_EV = 13.605693122994
_CHANNEL_NAMES = ("up", "down")


class SpinSectorsAbandoned(RuntimeError):
    """A spin-polarized sector loop failed a test of the module.

    ``stage`` names the loop (``the start Hamiltonian``, ``the SCF loop``,
    ``the semilocal start of the Fock loop``, ``the end of the Fock loop``),
    ``report`` what was found.
    """

    def __init__(self, stage: str, report: "FractionalOccupations") -> None:
        self.stage = stage
        self.report = report
        super().__init__(
            f"Spin_Symmetry_Sectors: {report.reason} after {stage} "
            f"({report.measure} = {report.largest:.3e} >= {report.tolerance:.0e}): "
            + "; ".join(report.lines)
        )


@dataclass(frozen=True)
class FractionalOccupations:
    """What a test of the module found: the largest value of its ``measure``, the tolerance and per-item lines."""

    largest: float
    tolerance: float
    lines: tuple[str, ...]
    reason: str = "fractional occupations"
    measure: str = "largest min(f, 1-f)"


def _multiplets(values: np.ndarray, width: float) -> list[list[int]]:
    """Runs of levels (indices, ascending) whose neighbours lie within ``width``."""

    order = np.argsort(values, kind="stable")
    groups: list[list[int]] = []
    for position in order:
        if groups and values[position] - values[groups[-1][-1]] <= width:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return groups


def sector_multiplets(
    values: np.ndarray,
    sectors: np.ndarray,
    *,
    width: float = START_MULTIPLET_WIDTH,
    spread: float = START_MULTIPLET_SPREAD,
) -> list[list[int]]:
    """The multiplets of one channel that the sectors constrain (indices, ascending).

    A run of levels whose neighbours lie within ``width`` and whose whole
    spread is at most ``spread`` (a longer run is a band: the unconverged top
    of a first subspace, the valence band of a large cluster), with levels in
    at least two sectors: partners within one sector mix there as freely as on
    the full grid.
    """

    sectors = np.asarray(sectors)
    return [
        group
        for group in _multiplets(values, width)
        if len(group) >= 2
        and float(np.ptp(values[group])) <= spread
        and np.unique(sectors[group]).size >= 2
    ]


def partly_filled_multiplets(
    eigenvalues,
    occupations,
    representations,
    *,
    fixed: bool,
    width: float = START_MULTIPLET_WIDTH,
    spread: float = START_MULTIPLET_SPREAD,
    tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE,
):
    """:class:`FractionalOccupations` of the partly filled multiplets of the channels, ``None`` where there are none.

    ``eigenvalues``, ``occupations`` and ``representations`` hold one array
    per channel (Ry; electrons per level; sector labels).  A multiplet of
    :func:`sector_multiplets` with ``d`` levels holding ``n`` electrons is
    partly filled when ``min(n, d - n) > tolerance``; with ``fixed``
    occupations (``occup.in``) one whose occupations are all equal is not (see
    the module).
    """

    largest = 0.0
    lines: list[str] = []
    for index, (values, filled, sectors) in enumerate(zip(eigenvalues, occupations, representations)):
        values = np.asarray(values, dtype=np.float64)
        filled = np.asarray(filled, dtype=np.float64)
        sectors = np.asarray(sectors, dtype=np.int64)
        for group in sector_multiplets(values, sectors, width=width, spread=spread):
            electrons = float(np.sum(filled[group]))
            part = min(electrons, len(group) - electrons)
            if part <= tolerance:
                continue
            if fixed and float(np.ptp(filled[group])) < 1.0e-12:
                continue
            largest = max(largest, part)
            levels = " ".join(f"{values[member] * _RYDBERG_TO_EV:.4f}" for member in group)
            labels = " ".join(str(int(sectors[member])) for member in group)
            lines.append(
                f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: {len(group)} levels within "
                f"{width:g} Ry at {levels} eV hold {electrons:.4f} electrons (sectors {labels})"
            )
    if not lines:
        return None
    return FractionalOccupations(
        largest=largest,
        tolerance=float(tolerance),
        lines=tuple(lines),
        reason="a partly filled multiplet",
        measure="largest min(n, d - n) of d levels holding n electrons",
    )


def _key(matrix: np.ndarray) -> tuple:
    return tuple(int(value) for value in np.rint(matrix).ravel())


def _generators(matrices, seed=()) -> list[np.ndarray]:
    """A subset of ``matrices`` (a group of signed permutations) that generates them together with ``seed``."""

    elements = {_key(np.eye(3)): np.eye(3)}
    generators: list[np.ndarray] = [np.rint(np.asarray(matrix, dtype=np.float64)) for matrix in seed]
    grown = bool(generators)
    while grown:
        grown = False
        for element in list(elements.values()):
            for generator in generators:
                product = element @ generator
                if _key(product) not in elements:
                    elements[_key(product)] = product
                    grown = True
    seeded = len(generators)
    for matrix in matrices:
        matrix = np.rint(np.asarray(matrix, dtype=np.float64))
        if _key(matrix) in elements:
            continue
        generators.append(matrix)
        grown = True
        while grown:
            grown = False
            for element in list(elements.values()):
                for generator in generators:
                    product = element @ generator
                    if _key(product) not in elements:
                        elements[_key(product)] = product
                        grown = True
    return generators[seeded:]


def _grid_images(grid, matrix: np.ndarray):
    """Rows of the images ``matrix r`` of the grid points; ``None`` where the operation does not map the grid onto itself."""

    shift = np.asarray(grid.settings.shift, dtype=np.float64)
    target = (np.asarray(grid.integer_coordinates, dtype=np.float64) + shift) @ np.asarray(matrix).T - shift
    rounded = np.rint(target)
    if np.max(np.abs(target - rounded), initial=0.0) > 1.0e-9:
        return None
    rows = grid.rows_for_integer_coordinates(rounded.astype(np.int64))
    if np.any(rows < 0):
        return None
    return rows


def atom_group_images(grid, atoms) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(matrix, rows)`` of the operations that test 2 of the module needs on ``grid``.

    The signed permutations that map the atoms (by species) and the grid onto
    themselves (the group of the full-grid Hamiltonian), reduced to a set that
    generates that group: the sign changes first (the sectors usually keep the
    densities invariant under them by construction; they are tested all the
    same, at a few operations' cost), then the others.  The grid is asked for
    every candidate: a domain or a grid shift with less symmetry than the
    atoms (a cylinder along y, a box with a shorter y side) keeps a subgroup,
    whose generators need not be among those of the group of the atoms.  An
    operation in a coset ``f H`` of a refused ``f`` is not asked again.
    ``rows`` are the grid rows of the images ``matrix r``.
    """

    from parsec_python.Forces.symmetry import atom_symmetry_operations

    def is_diagonal(matrix) -> bool:
        return np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0

    matrices = [np.rint(np.asarray(operation.matrix, dtype=np.float64)) for operation in atom_symmetry_operations(atoms)]
    # The sign changes first: the generators returned are those beyond their group.
    matrices.sort(key=lambda matrix: not is_diagonal(matrix))
    closure = {_key(np.eye(3)): np.eye(3)}
    generators: list[tuple[np.ndarray, np.ndarray]] = []
    refused: list[np.ndarray] = []
    for matrix in matrices:
        if _key(matrix) in closure:
            continue
        # ``matrix`` in ``f H`` for a refused ``f`` and the grid operations
        # ``H`` found so far does not map the grid either (else ``f`` would).
        if any(_key(other.T @ matrix) in closure for other in refused):
            continue
        rows = _grid_images(grid, matrix)
        if rows is None:
            refused.append(matrix)
            continue
        generators.append((matrix, rows))
        grown = True
        while grown:
            grown = False
            for element in list(closure.values()):
                for generator, _ in generators:
                    product = element @ generator
                    if _key(product) not in closure:
                        closure[_key(product)] = product
                        grown = True
    return generators


def density_symmetry_breaking(densities, grid, atoms=None, tolerance: float = DENSITY_SYMMETRY_TOLERANCE, images=None):
    """:class:`FractionalOccupations` of channel densities that break the point group of ``atoms``, else ``None``.

    Each operation of :func:`atom_group_images` (``images``, else built from
    ``grid`` and ``atoms``) is tested by ``h^3 sum |rho(g r) - rho(r)|`` in
    electrons, per channel.
    """

    if images is None:
        images = atom_group_images(grid, atoms)
    largest = 0.0
    lines: list[str] = []
    for matrix, rows in images:
        for index, density in enumerate(densities):
            values = np.asarray(density, dtype=np.float64)
            deviation = float(grid.volume_element * np.sum(np.abs(values[rows] - values)))
            largest = max(largest, deviation)
            if deviation >= tolerance:
                lines.append(
                    f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: the density differs by "
                    f"{deviation:.4f} electrons from its image under the operation "
                    f"{np.rint(matrix).astype(int).tolist()} of the atoms"
                )
    if largest < tolerance:
        return None
    return FractionalOccupations(
        largest=largest,
        tolerance=float(tolerance),
        lines=tuple(lines),
        reason="a channel density that breaks the point group of the atoms",
        measure="largest int |rho(g r) - rho(r)| (electrons)",
    )


def fractional_occupations(channels, tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE):
    """:class:`FractionalOccupations` of the spin ``channels`` of a result, ``None`` where all are integer.

    ``channels`` are :class:`parsec_python.models.SpinChannelResult` (or
    anything with ``eigenvalues`` in Ry, ``occupations`` and
    ``representations``).  Only a fractional occupation that the sectors
    constrain counts: one in a multiplet whose levels lie in two or more
    sectors, exact (levels within :data:`MULTIPLET_WIDTH` of each other) or of
    :func:`sector_multiplets`.  A fractional single level, or a multiplet within
    one sector, is what the full grid has as well.  The line of a multiplet
    gives its levels (eV), occupations and sectors.
    """

    largest = 0.0
    lines: list[str] = []
    for index, channel in enumerate(channels):
        occupations = np.asarray(channel.occupations, dtype=np.float64)
        if not occupations.size:
            continue
        values = np.asarray(channel.eigenvalues, dtype=np.float64)
        sectors = np.asarray(getattr(channel, "representations", np.zeros(values.size)), dtype=np.int64)
        fractional = np.minimum(occupations, 1.0 - occupations)
        groups = [
            group for group in _multiplets(values, MULTIPLET_WIDTH) if np.unique(sectors[group]).size >= 2
        ]
        exact = {tuple(group) for group in groups}
        groups += [group for group in sector_multiplets(values, sectors) if tuple(group) not in exact]
        for group in groups:
            if float(np.max(fractional[group])) < tolerance:
                continue
            largest = max(largest, float(np.max(fractional[group])))
            levels = " ".join(f"{values[member] * _RYDBERG_TO_EV:.4f}" for member in group)
            filled = " ".join(f"{occupations[member]:.4f}" for member in group)
            labels = " ".join(str(int(sectors[member])) for member in group)
            lines.append(
                f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: {len(group)} level(s) at {levels} eV "
                f"hold {float(np.sum(occupations[group])):.4f} electrons (occupations {filled}; sectors {labels})"
            )
    if largest < tolerance:
        return None
    return FractionalOccupations(largest=largest, tolerance=float(tolerance), lines=tuple(lines))


class _ChannelPair:
    """The two channels' orbit values as one vector, for a mixer of :mod:`.symmetry_fields`.

    The interface :class:`~.symmetry_fields.SymmetryAndersonMixer` and
    :class:`~.symmetry_fields.SymmetryWeightedMixer` read of a reducer:
    multiplicity weights of both halves, and plain arrays in and out.
    """

    def __init__(self, reducer: SymmetrySCFReducer) -> None:
        reduction = reducer.reduction
        self.size = 2 * int(reduction.wedge_size)
        weights = np.concatenate((reducer.weights, reducer.weights))
        weights.setflags(write=False)
        self.weights = weights
        self.reduction = SimpleNamespace(
            multiplicities=np.concatenate((reduction.multiplicities, reduction.multiplicities)),
            wedge_size=self.size,
        )

    def wedge_values(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.size,):
            raise ValueError("the two channels' orbit values do not match the wedge")
        return array

    def field(self, values) -> np.ndarray:
        return self.wedge_values(values)

    def expand_values(self, values) -> np.ndarray:
        return self.wedge_values(values)


class SpinSectorReducer:
    """The scalar-field adapter of a spin-polarized loop in the symmetry sectors.

    It is a :class:`~.symmetry_fields.SymmetrySCFReducer` (``reducer``) for the
    fields of one channel, and adds what
    :func:`parsec_python.SCF.spin_polarized.run_spin_scf` asks of a reduced
    loop: the two channels joined for the mixer (up first, as ``spinmix.f90``)
    and split again, PARSEC's residual norms per channel and the spin energy,
    each with the orbit multiplicities as quadrature weights, which gives the
    full-grid sums of invariant fields, and the symmetry guard (see the
    module).  ``hybrid`` says whether the run is a hybrid's Fock loop, which
    names the loop of a failed check.
    """

    spin_channels_supported = True

    def __init__(
        self,
        reducer: SymmetrySCFReducer,
        *,
        hybrid: bool,
        tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE,
        fixed_occupations: bool = False,
        structure=None,
    ) -> None:
        self.reducer = reducer
        self.hybrid = bool(hybrid)
        self.tolerance = float(tolerance)
        # Occupations of occup.in (no Fermi level): test 3 does not apply.
        self.fixed_occupations = bool(fixed_occupations)
        # ``structure()`` -> ``(grid, atoms)`` of the present structure, for test 2.
        self.structure = structure
        self._pair = _ChannelPair(reducer)
        # ``(iteration, asymmetry, criterion)`` of each test-2 measure at the convergence of a loop.
        self.symmetry_measures: list[tuple[int, float | None, float]] = []

    # ----- one channel: the SymmetrySCFReducer interface ------------------

    @property
    def reduction(self):
        return self.reducer.reduction

    def field(self, values):
        return self.reducer.field(values)

    def from_full(self, values):
        return self.reducer.from_full(values)

    def to_full(self, values) -> np.ndarray:
        return self.reducer.to_full(values)

    def wedge_values(self, values) -> np.ndarray:
        return self.reducer.wedge_values(values)

    def expand_values(self, values) -> np.ndarray:
        return self.reducer.expand_values(values)

    def minimum(self, values) -> float:
        return self.reducer.minimum(values)

    def maximum(self, values) -> float:
        return self.reducer.maximum(values)

    def weighted_dot(self, left, right) -> float:
        return self.reducer.weighted_dot(left, right)

    # ----- two channels ---------------------------------------------------

    def join_channels(self, fields) -> np.ndarray:
        """The orbit values of both channels, up first."""

        up, down = fields
        return np.concatenate((self.reducer.wedge_values(up), self.reducer.wedge_values(down)))

    def split_channels(self, values) -> list:
        """The two channel fields of a joined vector."""

        values = self._pair.wedge_values(values)
        half = self._pair.size // 2
        return [self.reducer.field(values[:half].copy()), self.reducer.field(values[half:].copy())]

    @property
    def weights(self) -> np.ndarray:
        return self.reducer.weights

    def join_blocks(self, fields) -> np.ndarray:
        """Several orbit fields end to end (a meta-GGA's ``V_up, V_down, vtau_up, vtau_down``)."""

        return self.reducer.join_blocks(fields)

    def split_blocks(self, values, count: int) -> list:
        return self.reducer.split_blocks(values, count)

    def block_mixer(self, settings, count: int):
        return self.reducer.block_mixer(settings, count)

    def spin_mixer(self, settings):
        """The mixer of ``settings.method`` for joined channel vectors (:meth:`join_channels`)."""

        if settings.method == "anderson":
            return SymmetryAndersonMixer(self._pair, settings)
        return SymmetryWeightedMixer(self._pair, make_mixer(settings, weights=self._pair.weights))

    def spin_potential_residual_metrics(
        self, input_potentials, output_potentials, densities, volume_element: float, electrons
    ) -> SpinResidualMetrics:
        """:func:`parsec_python.Mixer.spin_potential_residual_metrics` on the orbits."""

        if volume_element <= 0:
            raise ValueError("volume element must be positive")
        weights = self.reducer.weights
        weighted: list[float] = []
        plain: list[float] = []
        for input_potential, output_potential, density, count in zip(
            input_potentials, output_potentials, densities, electrons
        ):
            residual = self.reducer.wedge_values(output_potential) - self.reducer.wedge_values(input_potential)
            if count < EMPTY_CHANNEL_ELECTRONS:
                plain.append(0.0)
                weighted.append(0.0)
                continue
            squared = residual * residual
            plain.append(float(np.sqrt(volume_element * np.dot(weights, squared))))
            density_values = self.reducer.wedge_values(density)
            weighted.append(
                float(np.sqrt(volume_element * np.dot(weights * density_values, squared) / count))
            )
        return SpinResidualMetrics(
            weighted_by_spin=(weighted[0], weighted[1]),
            plain_by_spin=(plain[0], plain[1]),
        )

    def spin_total_energy(
        self,
        eigenvalues,
        occupations,
        densities,
        input_effective_potentials,
        ionic_potential,
        output_hartree_potential,
        output_xc_potentials,
        exchange_correlation_energy: float,
        ion_ion_energy: float,
        volume_element: float,
        hubbard_energy: float | None = None,
        alpha_z_energy: float | None = None,
    ):
        """:func:`parsec_python.Energy.spin_total_energy` with multiplicity-weighted orbit sums."""

        wedge = self.reducer.wedge_values
        weights = self.reducer.weights
        ionic = wedge(ionic_potential)
        band_energy = 0.0
        old_hxc_integral = 0.0
        vxc_integral = 0.0
        for values, filled, density, input_potential, xc_potential in zip(
            eigenvalues, occupations, densities, input_effective_potentials, output_xc_potentials
        ):
            values = np.asarray(values, dtype=float)
            filled = np.asarray(filled, dtype=float)
            if filled.shape != values.shape:
                raise ValueError("eigenvalues and occupations must have the same shape")
            weighted_density = weights * wedge(density)
            band_energy += float(np.dot(filled, values))
            old_hxc_integral += float(volume_element * np.dot(weighted_density, wedge(input_potential) - ionic))
            vxc_integral += float(volume_element * np.dot(weighted_density, wedge(xc_potential)))
        if alpha_z_energy is not None:
            band_energy += float(alpha_z_energy)
        total_density = weights * (wedge(densities[0]) + wedge(densities[1]))
        hartree_integral = float(volume_element * np.dot(total_density, wedge(output_hartree_potential)))
        electron_ion = float(volume_element * np.dot(total_density, ionic))
        electronic = float(band_energy - old_hxc_integral + 0.5 * hartree_integral + exchange_correlation_energy)
        return _breakdown(
            band_energy,
            hartree_integral,
            vxc_integral,
            exchange_correlation_energy,
            electron_ion,
            ion_ion_energy,
            electronic,
            hubbard_energy,
        )

    def spin_hooks(self) -> dict:
        """The hooks of :func:`parsec_python.SCF.spin_polarized.run_spin_scf` for this adapter."""

        return {
            "mixer_factory": self.spin_mixer,
            "residual_metrics_evaluator": self.spin_potential_residual_metrics,
            "total_energy_evaluator": self.spin_total_energy,
            "scalar_field_adapter": self,
        }

    # ----- the guard ------------------------------------------------------

    def _stage(self, final: bool) -> str:
        if final and self.hybrid:
            return "the end of the Fock loop"
        if self.hybrid:
            return "the semilocal start of the Fock loop"
        return "the SCF loop"

    #: Further solves of the start Hamiltonian that test 1 may ask for (see :meth:`start_spectrum_report`).
    start_spectrum_passes = START_SPECTRUM_PASSES
    #: Largest change (Ry) of the levels up to the Fermi level between two such solves that counts as converged.
    start_level_change = START_LEVEL_CHANGE

    def start_spectrum_report(self, eigenvalues, occupations, representations):
        """:func:`partly_filled_multiplets` of the start Hamiltonian's levels (test 1), ``None`` where there are none.

        :func:`parsec_python.SCF.spin_polarized.run_spin_scf` asks after the
        occupations of the first iteration of a loop that starts from the
        atomic densities (one array per channel each).  A report is only a
        candidate until the levels are converged: the loop solves the start
        Hamiltonian again, from its subspaces, up to
        :attr:`start_spectrum_passes` times, and calls :meth:`abandon_start`
        when the report stays and the levels up to the Fermi level moved by at
        most :attr:`start_level_change` in the last solve.  The levels of the
        first filter of a large cluster are not converged near its top (the
        nanodiamond C459H204: seven levels within 0.015 Ry around a first
        Fermi level of +8 eV), and a report on them disappears or keeps moving.
        """

        return partly_filled_multiplets(eigenvalues, occupations, representations, fixed=self.fixed_occupations)

    def abandon_start(self, report, passes: int = 0) -> None:
        """Raise :class:`SpinSectorsAbandoned` for a converged report of :meth:`start_spectrum_report`."""

        if passes:
            report = FractionalOccupations(
                largest=report.largest,
                tolerance=report.tolerance,
                lines=report.lines + (f"(the levels of the start Hamiltonian after {passes} further solve(s))",),
                reason=report.reason,
                measure=report.measure,
            )
        raise SpinSectorsAbandoned("the start Hamiltonian", report)

    def check_start_spectrum(self, eigenvalues, occupations, representations) -> None:
        """Raise :class:`SpinSectorsAbandoned` where the levels given have a partly filled multiplet (test 1)."""

        report = self.start_spectrum_report(eigenvalues, occupations, representations)
        if report is not None:
            self.abandon_start(report)

    #: Test 2 (see the module): tolerance, the bound above which no tightening is tried, and the tightenings.
    density_symmetry_tolerance = DENSITY_SYMMETRY_TOLERANCE
    density_symmetry_noise_bound = DENSITY_SYMMETRY_NOISE_BOUND
    density_symmetry_tightenings = DENSITY_SYMMETRY_TIGHTENINGS

    def _grid_and_images(self):
        """The grid of the present structure and the operations of test 2 on it (built once per structure)."""

        grid, atoms = self.structure()
        key = (id(grid), tuple(tuple(np.asarray(atom.position, dtype=float).tolist()) for atom in atoms))
        if getattr(self, "_images_key", None) != key:
            self._images = atom_group_images(grid, atoms)
            self._images_key = key
        return grid, self._images

    def density_symmetry_measure(self, densities) -> float | None:
        """The largest ``int |rho(g r) - rho(r)|`` (electrons) of the full-grid channel ``densities`` (test 2).

        ``None`` without a structure.  :func:`parsec_python.SCF.spin_polarized.run_spin_scf`
        asks at the convergence of a loop without an exchange operator and
        tightens the loop's criterion while the measure is above the
        tolerance but below :attr:`density_symmetry_noise_bound` and shrinks
        (see the module).
        """

        if self.structure is None:
            return None
        grid, images = self._grid_and_images()
        largest = 0.0
        for _, rows in images:
            for density in densities:
                values = np.asarray(density, dtype=np.float64)
                largest = max(largest, float(grid.volume_element * np.sum(np.abs(values[rows] - values))))
        return largest

    def check_spin_result(self, result, *, final: bool = False) -> None:
        """Raise :class:`SpinSectorsAbandoned` where ``result`` fails test 2 or 3 of the module.

        ``final``: the check of the whole run (for a hybrid, the state at the
        end of the Fock loop); otherwise that of one SCF loop without an
        exchange operator.
        """

        result = getattr(result, "result", result)
        channels = getattr(result, "spin", None)
        if channels is None:
            return
        if self.structure is not None:
            grid, images = self._grid_and_images()
            report = density_symmetry_breaking(
                [channel.density for channel in channels], grid, tolerance=DENSITY_SYMMETRY_TOLERANCE, images=images
            )
            if report is not None:
                raise SpinSectorsAbandoned(self._stage(final), report)
        if self.fixed_occupations:
            return
        report = fractional_occupations(channels, self.tolerance)
        if report is not None:
            raise SpinSectorsAbandoned(self._stage(final), report)


class HostSpinXCOnOrbits:
    """A host spin functional for the orbit fields of a sector loop.

    ``evaluate(up, down)`` is a full-grid spin evaluator
    (:meth:`PreparedSinglePointSystem.evaluate_spin_xc` of the present
    structure); both channels are expanded to the full grid, evaluated, and
    the potentials and the energy density reduced to one value per orbit,
    an exact representation change for invariant fields.
    """

    def __init__(self, evaluate, reducer: SymmetrySCFReducer, label: str) -> None:
        self.evaluate = evaluate
        self.reducer = reducer
        self.implementation_label = f"{label} on the full grid, from and to the symmetry orbits"

    def __call__(self, valence_up, valence_down) -> SpinXCResult:
        evaluated = self.evaluate(self.reducer.to_full(valence_up), self.reducer.to_full(valence_down))
        return SpinXCResult(
            potential_up=self.reducer.from_full(evaluated.potential_up),
            potential_down=self.reducer.from_full(evaluated.potential_down),
            energy_density=self.reducer.from_full(evaluated.energy_density),
            total_energy=evaluated.total_energy,
        )


__all__ = [
    "DENSITY_SYMMETRY_TOLERANCE",
    "FractionalOccupations",
    "START_MULTIPLET_WIDTH",
    "atom_group_images",
    "density_symmetry_breaking",
    "partly_filled_multiplets",
    "HostSpinXCOnOrbits",
    "MULTIPLET_WIDTH",
    "SPIN_SECTOR_OCCUPATION_TOLERANCE",
    "SpinSectorReducer",
    "SpinSectorsAbandoned",
    "fractional_occupations",
]
