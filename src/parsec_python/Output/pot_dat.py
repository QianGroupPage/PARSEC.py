"""PARSEC 1.5's ``pot.dat``: the frozen sample field of an FDET/AFM tip run.

``potsave.F90`` (PARSEC 1.5, ``io/potsave.F90``) writes the file after the
SCF loop of a run, and the FDET tip runs read it (``potential/potfield.F90``
and ``fdet_tipsphere.F90:447-560``).  It is a gfortran sequential
unformatted file (4-byte little-endian record markers; the record helpers of
:mod:`parsec_python.Output.parsec_dat`).  The records, in order:

1. ``newk`` (real8, Ry): ``eeig - ehxcold - eion`` of the last ``totnrg``,
   the kinetic plus nonlocal pseudopotential energy of the electrons
   (with FDET plus ``Integral v_nad rho``);
2. ``atom_num`` (int4); then one record ``x, y, z`` (3 real8, bohr) per atom;
3. ``stype(10)`` (int4): ``nspin``; 0 real, 1 complex, 2 spin-orbit;
   0 cluster, 2 wire, 3 slab, 1 bulk; zeros;
4. cluster: ``step(1), rmax`` (2 real8); periodic: three records ``step(3)``,
   ``box_size(3)``, ``latt_vec(3,3)``;
5. ``ndim`` | ``shift(3)`` | ``nwedge, ntrans`` | ``rmtrx`` | ``trans`` |
   ``tnp`` | ``alatt, invlat`` | ``chi`` | ``(kx,ky,kz)`` of the wedge points
   (the symmetry block of ``parsec.dat``);
6. five fields of ``nwedge`` real8: ``vion`` (the local ionic potential of
   the run, without the FDET field; with the potential of the point charges
   of a run without FDET), ``vhart``, ``vxc(:,1)``, ``rho(:,1)`` (the valence
   density), ``rhoc`` (the core density).

The layout was checked on a ``pot.dat`` of PARSEC 1.5 (20 records for the
two atoms of an H2 sample).  The readers of 1.5 use ``newk``, the grid
records, ``vion``, ``vhart`` and ``rho``; they skip ``vxc`` and ``rhoc``.

Python writes the full grid (``nwedge = ndim``, the identity alone), as for
``parsec.dat``, and only for an isolated, spin-unpolarized run.  Deviation
from 1.5: the file is written only when the input asks for it
(``Write_Potential_Field``), never by default, and a run never overwrites
the field file it reads (``Potential_Field_Name``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parsec_dat import (
    ParsecDatSymmetry,
    _read_symmetry,
    _values,
    _write_symmetry,
    cluster_alatt,
    fortran_integer_coordinates,
    grid_rmax,
    read_record,
    system_type,
    write_record,
)

#: The file name ``potsave.F90`` writes.
POT_DAT_NAME = "pot.dat"


class PotentialFieldError(ValueError):
    """A ``pot.dat`` that cannot be read or does not fit the run.

    ``code`` is the PARSEC 1.5 error number (``potfield.F90``: 611 the file
    cannot be opened, 612 a record cannot be read, 601 the grid spacing,
    602 the grid size, 603 the index range, 604 a grid point not found;
    ``fdet_tipsphere.F90``: 621-630), or 0 for a check of this port only.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{message} (PARSEC error {code})" if code else message)
        self.code = int(code)


@dataclass(frozen=True)
class PotentialFieldFile:
    """The records of one ``pot.dat`` (all fields in Ry, e/bohr^3, bohr).

    ``steps`` holds ``step(3)`` (a cluster file has one step, repeated);
    ``rmax`` is set for a cluster, ``box_size`` and ``lattice_vectors``
    (rows ``a1, a2, a3``, the columns of Fortran ``latt_vec``) for a
    periodic file.  ``points`` are the ``(kx,ky,kz)`` of the ``nwedge``
    wedge points, in the order of the five fields.
    """

    kinetic_energy: float
    atom_positions: np.ndarray
    system_type: np.ndarray
    steps: np.ndarray
    ndim: int
    shift: np.ndarray
    symmetry: ParsecDatSymmetry
    points: np.ndarray
    ionic_potential: np.ndarray
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    density: np.ndarray
    core_density: np.ndarray
    rmax: float | None = None
    box_size: np.ndarray | None = None
    lattice_vectors: np.ndarray | None = None

    @property
    def boundary_code(self) -> int:
        """``stype(3)``: 0 cluster, 1 bulk, 2 wire, 3 slab."""

        return int(self.system_type[2])

    @property
    def is_cluster(self) -> bool:
        return self.boundary_code == 0

    @property
    def nwedge(self) -> int:
        return int(self.points.shape[0])

    @property
    def ntrans(self) -> int:
        return self.symmetry.ntrans


def write_potential_field(path: str | Path, data: PotentialFieldFile) -> Path:
    """Write ``data`` in the record layout of ``potsave.F90``."""

    path = Path(path)
    with path.open("wb") as stream:
        write_record(stream, np.asarray([data.kinetic_energy], dtype="<f8"))
        positions = np.asarray(data.atom_positions, dtype=np.float64).reshape(-1, 3)
        write_record(stream, np.asarray([positions.shape[0]], dtype="<i4"))
        for position in positions:
            write_record(stream, np.asarray(position, dtype="<f8"))
        write_record(stream, np.asarray(data.system_type, dtype="<i4"))
        if data.is_cluster:
            write_record(stream, np.asarray([data.steps[0], data.rmax], dtype="<f8"))
        else:
            write_record(stream, np.asarray(data.steps, dtype="<f8"))
            write_record(stream, np.asarray(data.box_size, dtype="<f8"))
            # latt_vec(j,k) column k is the lattice vector k: rows of lattice_vectors.
            write_record(stream, np.asarray(data.lattice_vectors, dtype="<f8").ravel())
        write_record(stream, np.asarray([data.ndim], dtype="<i4"))
        write_record(stream, np.asarray(data.shift, dtype="<f8"))
        _write_symmetry(stream, data.symmetry, np.asarray(data.points))
        for values in (
            data.ionic_potential,
            data.hartree_potential,
            data.xc_potential,
            data.density,
            data.core_density,
        ):
            write_record(stream, np.asarray(values, dtype="<f8"))
    return path


def read_potential_field(path: str | Path) -> PotentialFieldFile:
    """Read a ``pot.dat`` of PARSEC 1.5 or of this port.

    Raises :class:`PotentialFieldError` 611 for a file that cannot be
    opened and 612 for one whose records cannot be read (``potfield.F90``).
    """

    path = Path(path)
    try:
        stream = path.open("rb")
    except OSError as error:
        raise PotentialFieldError(611, f"cannot open the FDET potential field file {path}: {error}") from error
    try:
        with stream:
            kinetic = float(_values(read_record(stream), "<f8", 1, "newk")[0])
            count = int(_values(read_record(stream), "<i4", 1, "atom_num")[0])
            if count < 0:
                raise ValueError("negative atom count")
            positions = np.array(
                [_values(read_record(stream), "<f8", 3, "atom") for _ in range(count)], dtype=np.float64
            ).reshape(count, 3)
            stype = _values(read_record(stream), "<i4", 10, "stype")
            rmax = box = lattice = None
            if int(stype[2]) == 0:
                step, rmax = (float(value) for value in _values(read_record(stream), "<f8", 2, "step,rmax"))
                steps = np.full(3, step)
            else:
                steps = _values(read_record(stream), "<f8", 3, "step")
                box = _values(read_record(stream), "<f8", 3, "box_size")
                lattice = _values(read_record(stream), "<f8", 9, "latt_vec").reshape(3, 3)
            ndim = int(_values(read_record(stream), "<i4", 1, "ndim")[0])
            shift = _values(read_record(stream), "<f8", 3, "shift")
            symmetry, points = _read_symmetry(stream)
            nwedge = points.shape[0]
            fields = [_values(read_record(stream), "<f8", nwedge, name) for name in ("vion", "vhart", "vxc", "rho", "rhoc")]
    except (EOFError, ValueError) as error:
        raise PotentialFieldError(612, f"failed while reading the FDET potential field file {path}: {error}") from error
    return PotentialFieldFile(
        kinetic_energy=kinetic,
        atom_positions=positions,
        system_type=stype.astype(np.int32),
        steps=np.asarray(steps, dtype=np.float64),
        ndim=ndim,
        shift=np.asarray(shift, dtype=np.float64),
        symmetry=symmetry,
        points=points,
        ionic_potential=fields[0],
        hartree_potential=fields[1],
        xc_potential=fields[2],
        density=fields[3],
        core_density=fields[4],
        rmax=None if rmax is None else float(rmax),
        box_size=box,
        lattice_vectors=lattice,
    )


def kinetic_energy_record(result, nonadditive_potential: np.ndarray | None = None) -> float:
    """``newk = pot%newke`` of ``totnrg.f90:149``: ``eeig - ehxcold - eion`` (Ry).

    ``eeig - integral rho V_in`` of the last iteration, plus
    ``integral rho v_nad`` for an FDET run, whose ``ehxcold`` leaves the
    non-additive kinetic potential out.
    """

    density = np.asarray(result.density, dtype=np.float64)
    volume = float(result.grid.volume_element)
    value = float(result.energies.eigenvalue) - volume * float(
        np.dot(density, np.asarray(result.input_effective_potential, dtype=np.float64))
    )
    if nonadditive_potential is not None:
        value += volume * float(np.dot(density, np.asarray(nonadditive_potential, dtype=np.float64)))
    return value


def potential_field_from_result(result, system) -> PotentialFieldFile:
    """The ``pot.dat`` of a finished isolated, spin-unpolarized SCF loop.

    ``system`` is the prepared system of ``result``; for an FDET run the
    sample field is taken out of its ionic potential and the non-additive
    kinetic potential out of ``ehxcold``, as ``parsec.F90:2223-2256`` does
    around ``potsave``.
    """

    grid = result.grid
    if getattr(result, "spin", None) is not None:
        raise ValueError("pot.dat of a spin-polarized run is not implemented")
    if hasattr(grid, "cell") or getattr(grid, "periodicity", None) is not None:
        raise ValueError("pot.dat is written for clusters only")
    ionic = np.asarray(system.ionic_potential, dtype=np.float64)
    field = getattr(system, "embedding", None)
    nonadditive = None
    if field is not None:
        # parsec.F90:2223-2227: vion - vsion - vshart around potsave
        ionic = ionic - field.ionic_potential - field.hartree_potential
        terms = getattr(result, "embedding", None)
        nonadditive = None if terms is None else terms.potential
    rmax = grid_rmax(grid)
    alatt = cluster_alatt(rmax)
    return PotentialFieldFile(
        kinetic_energy=kinetic_energy_record(result, nonadditive),
        atom_positions=np.array([atom.position for atom in result.atoms], dtype=np.float64),
        system_type=system_type(1, 0, 0),
        steps=np.full(3, float(grid.spacing)),
        ndim=int(grid.size),
        shift=np.asarray(grid.settings.shift, dtype=np.float64),
        symmetry=ParsecDatSymmetry.identity(alatt),
        points=fortran_integer_coordinates(grid),
        ionic_potential=ionic,
        hartree_potential=np.asarray(result.hartree_potential, dtype=np.float64),
        xc_potential=np.asarray(result.xc_potential, dtype=np.float64),
        density=np.asarray(result.density, dtype=np.float64),
        core_density=np.asarray(result.core_density, dtype=np.float64),
        rmax=rmax,
    )


def write_run_potential_field(directory: str | Path, result, system) -> Path:
    """``potsave`` after the last SCF loop: ``<directory>/pot.dat``."""

    return write_potential_field(Path(directory) / POT_DAT_NAME, potential_field_from_result(result, system))


def write_potential_field_after_run(translation, system, result, outcome, dynamics, write, directory) -> Path | None:
    """``Write_Potential_Field``: ``pot.dat`` of the last structure of a run.

    ``outcome`` and ``dynamics`` are the relaxation and molecular-dynamics
    results of the run (``None`` without), whose ``system`` is that of the
    last structure.  ``write`` takes the line of the run log.  Nothing is
    written without ``Write_Potential_Field``.
    """

    if not getattr(translation, "write_potential_field", False):
        return None
    last = system
    for ionic in (outcome, dynamics):
        if ionic is not None and getattr(ionic, "system", None) is not None:
            last = ionic.system
    path = write_run_potential_field(directory, result, last)
    write(f" pot.dat: {path} (frozen field of {int(result.grid.size)} points for FDET tip runs)")
    return path


__all__ = [
    "POT_DAT_NAME",
    "PotentialFieldError",
    "PotentialFieldFile",
    "kinetic_energy_record",
    "potential_field_from_result",
    "read_potential_field",
    "write_potential_field",
    "write_potential_field_after_run",
    "write_run_potential_field",
]
