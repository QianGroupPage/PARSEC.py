"""``Restart_Run``: the SCF loop starts from the potential of ``parsec.dat``.

A translation of ``restart_run.F90`` and of its caller (``parsec.F90``
l.917-1020).  PARSEC reads the grid of the file, checks it against the grid
of the run, maps the stored points onto the current ones and keeps the
stored self-consistent potential ``vnew`` (Ry) of each spin channel and the
stored densities.  It reads the eigenvalues and skips the occupations, but
never the wavefunctions: the number of computed states is reset to zero, so
the first SCF iteration solves the eigenproblem of the stored potential from
scratch (CHEBFF/CHEBDAV as in a first run) and the mixer starts with an
empty history.  Only when the sphere radius changed (``ixtrpflg = 1``) are
the Hartree and exchange-correlation potentials rebuilt from the stored
density, which is zero outside the stored sphere.

This module returns that start as an :class:`SCFWarmStart` without an
eigensolver state, which :func:`parsec_python.SCF.run_scf` (and the spin
loop it dispatches to) takes as the potential of its first iteration.

Deviations from PARSEC, all of which only refuse or repair what PARSEC would
silently get wrong:

* The file may be symmetry-reduced (PARSEC writes the irreducible wedge of
  its Abelian subgroup).  Python works on the full grid and unfolds the wedge
  with the operations stored in the file
  (:func:`parsec_python.Output.parsec_dat.grid_rows_of_file`), so the number
  of representations and the character table of the file (errors 369 and
  370 of PARSEC) need not match the run.
* PARSEC interprets the stored integer coordinates with the grid shift of
  the current run and never reads the stored shift.  Here a different shift
  (for instance a file of a run with another ``Ignore_Symmetry``) is an
  error, and so is a cluster spacing that differs by more than 1e-10 bohr
  (PARSEC compares the two doubles exactly, error 366).
* With k-points PARSEC skips six records instead of three for a stored
  k-point that differs from the current one, which desynchronizes the file.
  The eigenvalues are not used here, so all k-points are read in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..Output.parsec_dat import (
    ParsecDat,
    boundary_code,
    grid_rmax,
    grid_rows_of_file,
    grid_shift,
    grid_steps,
    read_parsec_dat,
)
from .single_point import SCFWarmStart

#: ``tol`` of ``restart_run.F90`` for the spacing and the cell vectors (bohr).
RESTART_TOLERANCE = 1.0e-10


class RestartError(ValueError):
    """A ``parsec.dat`` that cannot restart this run.

    ``code`` is the ``ierr`` of ``restart_run.F90`` for the same mistake,
    ``None`` for a check that PARSEC does not make.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message if code is None else f"{message} (PARSEC error {code})")
        self.code = code


@dataclass(frozen=True)
class RestartFields:
    """What a restart takes from ``parsec.dat``, on the grid of the run.

    ``potentials`` holds ``vnew`` of each spin channel (Ry) and
    ``densities`` the density of each channel (the total density for one
    channel), in e/bohr^3, zero at points the file does not hold.
    ``extrapolate`` is ``ixtrpflg``: the sphere radius changed, so the
    potentials are to be rebuilt from the densities.  ``missing_points``
    counts grid points without a stored value, ``dropped_points`` stored
    points outside the domain of the run, ``file_points`` and
    ``file_operations`` the points and symmetry operations of the file.
    ``stored_states`` is the smallest number of levels the file holds for a
    channel and k-point (``nec`` of ``restart_run.F90``), ``None`` where it
    is not known.
    """

    date_label: str
    potentials: tuple[np.ndarray, ...]
    densities: tuple[np.ndarray, ...]
    extrapolate: bool
    stored_radius: float
    missing_points: int
    dropped_points: int
    file_points: int
    file_operations: int
    stored_states: int | None = None

    @property
    def total_density(self) -> np.ndarray:
        """``rho(:,1)``: the sum of the channels with two."""

        if len(self.densities) == 1:
            return self.densities[0]
        return self.densities[0] + self.densities[1]


def _is_periodic(grid) -> bool:
    return hasattr(grid, "points_per_axis")


def check_restart_file(
    data: ParsecDat,
    grid,
    *,
    spin_count: int,
    complex_wavefunctions: bool,
    spinors: bool = False,
) -> bool:
    """The sanity checks of ``restart_run.F90``; returns ``ixtrpflg``.

    Raises :class:`RestartError` with PARSEC's error number where the file
    belongs to another kind of calculation or another grid.  ``spinors``
    marks a self-consistent spin-orbit run: its file must be one of a
    spin-orbit run (``stype(2) = 2``, which sets ``is_so`` in
    ``restart_run.F90``; two spin components).  A non-collinear file
    (``stype(2)`` 3 or 4) restarts nothing: PARSEC does not store the
    direction of its magnetization (``spin3d``), and ``restart_run.F90`` reads
    such a file as one without spinors.
    """

    if spinors:
        if data.wavefunction_kind != 2:
            raise RestartError(
                "a self-consistent spin-orbit run restarts from the parsec.dat of "
                "a spin-orbit run (stype(2) = 2); this file has stype(2) = "
                f"{data.wavefunction_kind}",
                362,
            )
        spin_count = 2
    if data.spin_count != spin_count:
        raise RestartError(
            f"parsec.dat holds {data.spin_count} spin channel(s), this run has "
            f"{spin_count}",
            361,
        )
    if data.wavefunction_kind in (3, 4):
        raise RestartError(
            "parsec.dat holds the spinors of a non-collinear run, whose "
            "magnetization direction PARSEC does not store (spin3d); a restart "
            "from it is not implemented",
            362,
        )
    if data.wavefunction_kind == 2 and not spinors:
        raise RestartError(
            "parsec.dat holds the spinors of a spin-orbit run; it restarts a "
            "self-consistent spin-orbit run (SO_from_scratch) only",
            362,
        )
    if spinors:
        pass
    elif complex_wavefunctions != data.is_complex:
        raise RestartError(
            "parsec.dat holds "
            + ("complex" if data.is_complex else "real")
            + " wavefunctions (k-points), this run "
            + ("complex" if complex_wavefunctions else "real")
            + " ones",
            362,
        )
    periodic = _is_periodic(grid)
    confined = bool(getattr(grid, "is_confined_periodic", False))
    expected_boundary = boundary_code(grid)
    if data.boundary_code != expected_boundary:
        names = {0: "cluster", 1: "bulk", 2: "wire", 3: "slab"}
        raise RestartError(
            "parsec.dat belongs to a "
            f"{names.get(data.boundary_code, str(data.boundary_code))} calculation, "
            f"this run is a {names[expected_boundary]} one",
            363,
        )
    steps = grid_steps(grid)
    if periodic:
        if np.max(np.abs(np.asarray(data.step) - steps)) > RESTART_TOLERANCE:
            raise RestartError(
                f"the grid spacing of parsec.dat {np.asarray(data.step).tolist()} "
                f"differs from that of this run {steps.tolist()}",
                364,
            )
        stored = np.asarray(data.periodic.lattice_vectors)
        current = np.asarray(grid.cell.lattice_vectors, dtype=float)
        if np.max(np.abs(stored - current)) > RESTART_TOLERANCE:
            raise RestartError(
                "the cell of parsec.dat differs from the cell of this run", 365
            )
    elif abs(float(data.step[0]) - float(steps[0])) > RESTART_TOLERANCE:
        raise RestartError(
            f"the grid spacing of parsec.dat ({float(data.step[0]):.12g} bohr) "
            f"differs from that of this run ({float(steps[0]):.12g} bohr)",
            366,
        )
    shift = grid_shift(grid)
    if np.max(np.abs(np.asarray(data.shift) - shift)) > 1.0e-12:
        raise RestartError(
            f"the grid shift of parsec.dat {np.asarray(data.shift).tolist()} "
            f"differs from that of this run {shift.tolist()}; PARSEC's grid shift "
            "is zero with Ignore_Symmetry and one half otherwise, so restart "
            "with the Ignore_Symmetry of the run that wrote the file"
        )
    radius = grid_rmax(grid)
    if periodic and not confined:
        # restart_run.F90: rmax = rmaxold for a bulk cell; a wire or slab
        # compares its radius (l.265) as a cluster does.
        return False
    same_radius = abs(radius - float(data.rmax)) <= 1.0e-12 * max(1.0, radius)
    if same_radius and data.ndim != grid.size:
        raise RestartError(
            "despite the same grid spacing and sphere radius, parsec.dat holds "
            f"{data.ndim} grid points and this run {grid.size}",
            367,
        )
    if radius > float(data.rmax) and data.ndim > grid.size:
        raise RestartError(
            "despite a larger sphere, this run has fewer grid points "
            f"({grid.size}) than parsec.dat ({data.ndim})",
            368,
        )
    return not same_radius


def restart_fields(
    data: ParsecDat,
    grid,
    *,
    spin_count: int,
    complex_wavefunctions: bool = False,
    spinors: bool = False,
) -> RestartFields:
    """Check ``data`` and put its potentials and densities on ``grid``.

    A spin-orbit file gives the potentials and densities of its two spin
    components (``vold(:,1:2)``, ``rho(:,2:3)``).
    """

    extrapolate = check_restart_file(
        data,
        grid,
        spin_count=spin_count,
        complex_wavefunctions=complex_wavefunctions,
        spinors=spinors,
    )
    if spinors:
        spin_count = 2
    mapping = grid_rows_of_file(data, grid)
    if not extrapolate and mapping.missing:
        raise RestartError(
            f"{mapping.missing} points of this grid have no value in parsec.dat "
            "although the spacing and the sphere radius are the same",
            367,
        )
    return RestartFields(
        date_label=data.date_label,
        potentials=tuple(mapping.field(values) for values in data.potentials[:spin_count]),
        densities=tuple(mapping.field(values) for values in data.densities[:spin_count]),
        extrapolate=bool(extrapolate),
        stored_radius=float(data.rmax),
        missing_points=mapping.missing,
        dropped_points=mapping.dropped,
        file_points=data.nwedge,
        file_operations=data.symmetry.ntrans,
        stored_states=min(
            (int(block.energies.size) for channel in data.levels for block in channel),
            default=None,
        ),
    )


def check_fixed_state_count(stored_states: int | None, number_of_states: int) -> None:
    """``Fix_Number_Eigenvalues``: the restart file must hold ``States_Num`` levels.

    PARSEC then keeps the stored number of eigenpairs (``restart_run.F90``
    and ``restart.F90`` read ``min(stored, nstate)`` of them) in every
    solve.  With fewer than ``States_Num`` the final test of ``eigval.F90``
    (l.700-716, ierr 404) is meant to stop the run; the Fortran binary
    instead repeats its diagonalization loop without work and crashes.
    """

    if stored_states is None or stored_states >= int(number_of_states):
        return
    raise RestartError(
        "Fix_Number_Eigenvalues: not enough eigenstates are computed! "
        f"{int(number_of_states)} eigenstates requested but only {int(stored_states)} "
        "found in the restart file (keep States_Num of the run that wrote it, "
        "or remove Fix_Number_Eigenvalues)",
        404,
    )


def restart_warm_start(system, fields: RestartFields) -> SCFWarmStart:
    """The start of the SCF loop of ``system`` from ``fields`` (``parsec.F90``).

    Without extrapolation the stored ``vnew`` is the potential of the first
    eigenproblem.  With it (``ixtrpflg = 1``) the potential is
    ``V_ion + V_H[rho] + V_xc[rho]`` of the stored densities, as
    ``rho_har`` makes it for a first run.  The starting vector of the first
    Poisson solve is ``-V_ion``, as for a first run; it does not change the
    solution.  No eigensolver state: the first iteration solves from
    scratch, the mixer starts empty.
    """

    ionic = np.asarray(system.ionic_potential, dtype=np.float64)
    spin = len(fields.potentials) == 2
    if not fields.extrapolate:
        potential = (
            tuple(np.array(value, dtype=np.float64) for value in fields.potentials)
            if spin
            else np.array(fields.potentials[0], dtype=np.float64)
        )
        return SCFWarmStart(input_potential=potential, hartree_potential=-ionic)
    if not spin:
        # run_scf adds V_H and V_xc of the guess and subtracts the reference.
        return SCFWarmStart(
            input_potential=ionic.copy(),
            hartree_potential=-ionic,
            density_guess=np.array(fields.densities[0], dtype=np.float64),
            hxc_reference=np.zeros_like(ionic),
        )
    # A loop that keeps its fields on symmetry orbits (the spin channels of
    # the accelerated driver in the sectors) evaluates them there; the warm
    # start holds full-grid fields.
    adapter = getattr(system, "scalar_field_adapter", None)
    inward = (lambda values: values) if adapter is None else adapter.from_full
    outward = (
        (lambda values: values)
        if adapter is None
        else (lambda values: adapter.to_full(values))
    )
    hartree = system.solve_hartree(
        inward(fields.total_density), initial_potential=inward(-ionic)
    )
    xc = system.evaluate_spin_xc(inward(fields.densities[0]), inward(fields.densities[1]))
    return SCFWarmStart(
        input_potential=tuple(
            ionic + outward(hartree.potential) + outward(potential) for potential in xc.potentials
        ),
        hartree_potential=np.asarray(outward(hartree.potential), dtype=np.float64),
    )


def restart_report_lines(fields: RestartFields) -> list[str]:
    """What ``restart_run.F90`` writes to ``parsec.out``, plus what Python did."""

    lines = [
        "",
        " Restarting from previous run",
        " ----------------------------",
        "",
        f"  parsec.dat file created on   {fields.date_label}",
        "",
    ]
    if fields.extrapolate:
        lines += [
            "",
            " WARNING: Extrapolating an initial guess based on",
            f" stored data for a sphere of radius {fields.stored_radius:12.6f} .",
            "",
        ]
    if fields.file_operations > 1:
        lines.append(
            f" parsec.dat holds the irreducible wedge of {fields.file_points} points "
            f"and {fields.file_operations} symmetry operations; Python unfolds it "
            "onto the full grid"
        )
    if fields.missing_points:
        lines.append(
            f" {fields.missing_points} grid points lie outside the stored domain: "
            "their stored density and potential are zero"
        )
    if fields.dropped_points:
        lines.append(
            f" {fields.dropped_points} stored points lie outside the current domain "
            "and are dropped"
        )
    lines.append(
        " The first SCF iteration solves the eigenproblem of the stored potential "
        "from scratch; the stored wavefunctions are not read (restart_run.F90)"
    )
    return lines


def load_restart(
    path: str | Path,
    system,
    *,
    spin_count: int | None = None,
    complex_wavefunctions: bool = False,
    spinors: bool = False,
) -> tuple[RestartFields, SCFWarmStart]:
    """Read ``path`` and return its fields and the warm start of ``system``.

    ``spinors``: a self-consistent spin-orbit run; the warm start then holds
    the stored potential of each spin component (the isolated spin-orbit
    loops take them as their first potential, see :func:`spin_orbit_start`).
    """

    source = Path(path)
    if not source.is_file():
        raise RestartError(f"Restart_Run: {source} not found", 371)
    if spin_count is None:
        spin_count = 2 if getattr(system.input.scf, "spin_polarized", False) else 1
    data = read_parsec_dat(source, wavefunctions=False)
    fields = restart_fields(
        data,
        system.grid,
        spin_count=spin_count,
        complex_wavefunctions=complex_wavefunctions,
        spinors=spinors,
    )
    if spinors:
        if fields.extrapolate:
            raise RestartError(
                "a spin-orbit restart from the parsec.dat of another sphere radius "
                "is not implemented"
            )
        ionic = np.asarray(system.ionic_potential, dtype=np.float64)
        return fields, SCFWarmStart(
            input_potential=tuple(np.array(value) for value in fields.potentials),
            hartree_potential=-ionic,
        )
    return fields, restart_warm_start(system, fields)


def spin_orbit_start(warm_start: SCFWarmStart, spin_polarized: bool):
    """The ``initial_potential`` of a spin-orbit loop from a spinor restart.

    The spin-polarized loop takes the potentials of the two components; the
    loop with one potential for both takes their average (equal for a
    file of that loop).
    """

    up, down = warm_start.input_potential
    if spin_polarized:
        return (np.asarray(up), np.asarray(down))
    return 0.5 * (np.asarray(up) + np.asarray(down))


# ----------------------------------------------------------------------------
# Restart_From_Wfndat: the legacy wfn.dat (restart.F90)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WfnDatRestart:
    """What a restart takes from ``wfn.dat``, on the grid of the run.

    As :class:`RestartFields`, plus ``wavefunctions``/``eigenvalues`` per
    channel: the stored states kept (``min(nstate, States_Num)`` of
    ``restart.F90``) on the full grid, zero where the file has no point.
    """

    fields: RestartFields
    wavefunctions: tuple[np.ndarray, ...]
    eigenvalues: tuple[np.ndarray, ...]


def check_wfn_dat(data, grid) -> bool:
    """The checks of ``restart.F90`` (errors 321-329); returns ``ixtrpflg``.

    This port works on the full grid with one representation, so the file
    must hold one representation with the character ``1`` (PARSEC compares
    ``nrep`` and ``chi`` with those of its run: 328, 329).  Spacings and
    cells are compared to ``1e-10`` bohr (PARSEC: exact equality).
    """

    periodic = _is_periodic(grid)
    if data.periodic != periodic:
        raise RestartError("wfn.dat belongs to another boundary condition than this run")
    steps = grid_steps(grid)
    if periodic:
        if np.max(np.abs(np.asarray(data.step) - steps)) > RESTART_TOLERANCE:
            raise RestartError(
                "stored grid spacing conflicts with current grid spacing", 321
            )
        sizes = np.linalg.norm(np.asarray(grid.cell.lattice_vectors, dtype=float), axis=1)
        if np.max(np.abs(np.asarray(data.box_size) - sizes)) > RESTART_TOLERANCE:
            raise RestartError(
                "the size of the supercell conflicts with the stored one", 322
            )
        if data.ndim != grid.size or data.nwedge != grid.size:
            raise RestartError(
                "despite the same grid spacing and cell size, the stored "
                "Hamiltonian size is incompatible with the current one",
                325,
            )
        extrapolate = False
    else:
        radius = grid_rmax(grid)
        if abs(float(data.step[0]) - float(steps[0])) > RESTART_TOLERANCE:
            raise RestartError(
                "stored grid spacing conflicts with current grid spacing", 323
            )
        if float(data.rmax) > radius + 1.0e-12 * max(1.0, radius):
            raise RestartError(
                "stored sphere size larger than current one; this case is not "
                "implemented",
                324,
            )
        same = abs(float(data.rmax) - radius) <= 1.0e-12 * max(1.0, radius)
        if same and (data.ndim != grid.size or data.nwedge != grid.size):
            raise RestartError(
                "despite the same grid spacing and sphere size, the stored "
                "Hamiltonian size is incompatible with the current one",
                326,
            )
        if not same and (data.ndim > grid.size or data.nwedge > grid.size):
            raise RestartError(
                "despite the same grid spacing and a larger sphere size, the "
                "stored Hamiltonian size is larger than the current one",
                327,
            )
        extrapolate = not same
    if data.representation_count != 1:
        raise RestartError(
            f"number of irreducible representations does not match "
            f"({data.representation_count} in wfn.dat, 1 on the full grid of "
            "this port)",
            328,
        )
    if np.any(np.asarray(data.characters) != 1):
        raise RestartError("character table does not match", 329)
    return extrapolate


def _subspace_start(system, vectors: np.ndarray, eigenvalues: np.ndarray):
    """An eigensolver state whose subspace begins with the stored orbitals.

    ``restart.F90`` loads the stored wavefunctions and sets ``eig_init``,
    so the first eigenproblem is a SUBSPACE filter of them (``eval_loc`` =
    the stored levels).  The state has the working size of the run; columns
    beyond the stored states are random vectors orthogonal to them, with the
    highest stored level as their value.
    """

    from ..Eigensolvers import EigvalState
    from ..Eigensolvers.subspace import SubspaceState
    from .single_point import _number_of_states

    requested = _number_of_states(system)
    size = int(system.grid.size)
    working = min(size, requested + int(system.input.eigensolver.working_buffer))
    stored = np.asarray(vectors, dtype=np.float64)[:, :working]
    values = np.asarray(eigenvalues, dtype=np.float64)[: stored.shape[1]]
    if stored.shape[1] < working:
        rng = np.random.default_rng(0)
        fill = rng.standard_normal((size, working - stored.shape[1]))
        fill -= stored @ (stored.T @ fill)
        stored = np.column_stack([stored, fill])
        top = float(values.max()) if values.size else 0.0
        values = np.concatenate([values, np.full(working - values.size, top)])
    basis, _ = np.linalg.qr(stored)
    # Keep the sign and order of the stored orbitals (QR only normalizes them
    # and makes the fill orthogonal).
    signs = np.sign(np.sum(basis * stored, axis=0))
    basis = basis * np.where(signs == 0.0, 1.0, signs)[None, :]
    return EigvalState(
        operator_dimension=size,
        requested_states=requested,
        working_states=working,
        initial_method=system.input.eigensolver.method,
        subspace=SubspaceState(
            operator_dimension=size,
            working_states=working,
            eigenvalues=values,
            vectors=basis,
            first_filter=True,
        ),
        solves_completed=1,
    )


def load_wfndat_restart(
    path: str | Path,
    system,
    *,
    spin_count: int | None = None,
) -> tuple[WfnDatRestart, SCFWarmStart]:
    """``Restart_From_Wfndat``: read ``wfn.dat`` and the warm start of ``system``.

    ``restart.F90`` reads ``vold`` and ``rho`` of every channel and the
    wavefunctions of ``min(nstate, States_Num)`` states, maps the stored
    points onto the current grid (``indexw`` of ``kx, ky, kz``) and, for a
    larger sphere, rebuilds the Hartree and exchange-correlation potentials
    from the stored density (``ixtrpflg = 1``).  The first eigenproblem then
    filters the stored states (:func:`_subspace_start`); the mixer starts
    empty.
    """

    from ..Output.parsec_dat import fortran_integer_coordinates
    from ..Output.wfn_dat import read_wfn_dat
    from .single_point import _number_of_states

    source = Path(path)
    if not source.is_file():
        raise RestartError(f"Restart_From_Wfndat: {source} not found")
    if spin_count is None:
        spin_count = 2 if getattr(system.input.scf, "spin_polarized", False) else 1
    grid = system.grid
    data = read_wfn_dat(
        source, periodic=_is_periodic(grid), spin_count=spin_count, complex_values=False
    )
    extrapolate = check_wfn_dat(data, grid)
    rows = {
        tuple(int(value) for value in point): row
        for row, point in enumerate(fortran_integer_coordinates(grid))
    }
    targets = np.array([rows.get(tuple(int(v) for v in point), -1) for point in data.points])
    present = targets >= 0
    dropped = int(np.count_nonzero(~present))
    missing = int(grid.size - np.count_nonzero(present))
    if not extrapolate and missing:
        raise RestartError(
            f"{missing} points of this grid have no value in wfn.dat although the "
            "spacing and the domain are the same"
        )

    def on_grid(values):
        values = np.asarray(values)
        shape = (grid.size,) + values.shape[1:]
        result = np.zeros(shape, dtype=values.dtype)
        result[targets[present]] = values[present]
        return result

    keep = _number_of_states(system)
    fields = RestartFields(
        date_label=data.date_label,
        potentials=tuple(on_grid(channel.potential) for channel in data.channels),
        densities=tuple(on_grid(channel.density) for channel in data.channels),
        extrapolate=bool(extrapolate),
        stored_radius=float(data.rmax) if data.rmax is not None else 1.0,
        missing_points=missing,
        dropped_points=dropped,
        file_points=data.nwedge,
        file_operations=1,
    )
    wavefunctions = tuple(on_grid(channel.wavefunctions[:, :keep]) for channel in data.channels)
    eigenvalues = tuple(np.asarray(channel.energies[:keep]) for channel in data.channels)
    restart = WfnDatRestart(fields=fields, wavefunctions=wavefunctions, eigenvalues=eigenvalues)
    start = restart_warm_start(system, fields)
    states = tuple(
        _subspace_start(system, vectors, values)
        for vectors, values in zip(wavefunctions, eigenvalues)
    )
    from dataclasses import replace

    start = replace(start, eigensolver_state=states[0] if spin_count == 1 else states)
    return restart, start


def wfndat_report_lines(restart: WfnDatRestart) -> list[str]:
    """What ``restart.F90`` writes to ``parsec.out``, plus what Python did."""

    fields = restart.fields
    lines = [
        "",
        " Restarting from previous run",
        " ----------------------------",
        "",
        f"  wfn.dat file created on   {fields.date_label}",
        "",
    ]
    if fields.extrapolate:
        lines += [
            "",
            " WARNING: Extrapolating an initial guess based on",
            " stored data for a smaller sphere.",
            "",
        ]
    if fields.dropped_points:
        lines.append(
            f" {fields.dropped_points} stored points lie outside the current domain "
            "and are dropped"
        )
    counts = ", ".join(str(vectors.shape[1]) for vectors in restart.wavefunctions)
    lines.append(
        f" The first SCF iteration filters the {counts} stored wavefunctions "
        "(restart.F90 loads them)"
    )
    return lines


def with_first_warm_start(backend, warm_start: SCFWarmStart):
    """An :class:`parsec_python.Relaxation.backend.IonicBackend` whose first
    SCF loop without a warm start takes ``warm_start`` (a restarted
    relaxation, molecular dynamics run or force calculation)."""

    from dataclasses import replace

    pending = [warm_start]
    run_scf = backend.run_scf

    def first_restarted(system, *args, warm_start=None, **keywords):
        start, pending[0] = pending[0], None
        return run_scf(
            system,
            *args,
            warm_start=start if warm_start is None else warm_start,
            **keywords,
        )

    return replace(backend, run_scf=first_restarted)


__all__ = [
    "RESTART_TOLERANCE",
    "RestartError",
    "RestartFields",
    "WfnDatRestart",
    "check_restart_file",
    "check_wfn_dat",
    "load_restart",
    "load_wfndat_restart",
    "restart_fields",
    "restart_report_lines",
    "restart_warm_start",
    "spin_orbit_start",
    "wfndat_report_lines",
    "with_first_warm_start",
]
