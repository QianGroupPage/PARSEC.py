"""Band structure along a k-path after the SCF (Fortran ``calc_bands``).

Like Fortran PARSEC, the band structure is a non-self-consistent
diagonalization in the converged self-consistent potential:

* the path is a list of segments in relative (reciprocal-lattice) coordinates;
  the shortest segment gets ``bandstruc_points`` k-points and the others
  proportionally more (:func:`~parsec_python.Grid.kpoints.band_path_kpoints`);
* at every k-point the Bloch Hamiltonian (kinetic with ``-2i k.grad``,
  Bloch-phase nonlocal and spin-orbit projectors) is built from the SCF
  potential and diagonalized from scratch by the SCF's *first* eigensolver,
  CHEBFF (``solver%eig_init = .false.`` in ``calc_bands``), i.e.
  ``FF_MaxIter`` Chebyshev filter cycles of degree ``Chebdav_Degree``.  CHEBFF
  has no convergence test; raise ``FF_MaxIter`` for tighter bands.  The largest
  residual ``|H v - e v|`` is reported for every k-point as a diagnostic.

Extension beyond Fortran: CHEBDAV adds ``Matvec_Blocksize`` vectors per Davidson
step, and at a highly symmetric k-point whose degenerate multiplet is larger than
the block (an 8-fold level in a folded cubic cell, 16-fold with spin-orbit) it can
leave reported bands unconverged, at a block size that is not predictable.  When a
reported band's residual exceeds ``residual_tolerance`` the k-point is re-solved
with a larger block and a new random start (up to ``max_attempts`` in total) and
the attempt with the smallest residual is kept; the number of attempts per k-point
is recorded in ``BandStructureResult.attempts``.

The Hamiltonian follows the SCF mode, so the same ``bandstruc`` block gives
scalar, collinear spin-polarized (up/down), spin-orbit, spin-orbit +
collinear-spin and non-collinear bands.  Comparing runs with and without
``SO_PSP``/``SO_from_scratch`` gives the effect of spin-orbit coupling.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..Eigensolvers import solve_eigval
from ..Grid import band_path_kpoints, explicit_kpoints
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..models import BandStructureResult, BandStructureSettings
from ..Parallel import make_pool
from .kpoints_soc import _eigval_settings, _k_point_operators, _spinor_hamiltonian

MODES = ("scalar", "spin_polarized", "soc", "soc_spin_polarized", "noncollinear")


def band_mode(problem) -> str:
    """Hamiltonian family of the SCF described by ``problem``."""
    scf = problem.scf
    has_soc_species = any(
        specification.spin_orbit for specification in problem.pseudopotentials.values()
    )
    if scf.noncollinear:
        return "noncollinear"
    if scf.self_consistent_spin_orbit:
        return "soc_spin_polarized" if scf.spin_polarized else "soc"
    if scf.spin_polarized:
        if has_soc_species:
            raise ValueError(
                "bands for the perturbative spin-orbit route (Spin_Polarization "
                "with SO_PSP but no SO_from_scratch) are not supported; use "
                "SO_from_scratch for spin-orbit bands, or SO_PSP: .false. for "
                "scalar-relativistic spin-polarized bands"
            )
        return "spin_polarized"
    return "scalar"


def band_potentials(result, mode: str) -> dict[str, np.ndarray]:
    """The converged local potentials a band calculation needs, from an SCF result."""
    if mode == "scalar":
        return {"effective_potential": np.asarray(result.output_effective_potential)}
    potentials = getattr(result, "scf_potentials", None)
    if not potentials:
        raise ValueError(f"the SCF result carries no converged potentials for mode {mode!r}")
    return {name: np.asarray(value) for name, value in potentials.items()}


_REQUIRED = {
    "scalar": ("effective_potential",),
    "spin_polarized": ("potential_up", "potential_down"),
    "soc": ("effective_potential",),
    "soc_spin_polarized": ("effective_potential", "xc_delta"),
    "noncollinear": ("effective_potential", "xc_field"),
}


def band_path_cartesian(
    settings: BandStructureSettings, cell
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Cartesian k-points (1/bohr) of every segment and the segment names."""
    starts = np.asarray([segment.start for segment in settings.segments])
    ends = np.asarray([segment.end for segment in settings.segments])
    start_cartesian, _ = explicit_kpoints(
        starts, unit="reciprocal_lattice_vectors", cell=cell
    )
    end_cartesian, _ = explicit_kpoints(ends, unit="reciprocal_lattice_vectors", cell=cell)
    segments = band_path_kpoints(start_cartesian, end_cartesian, settings.points)
    return segments, tuple(segment.name for segment in settings.segments)


def load_scf_for_bands(archive_path, problem, system, mode: str):
    """Converged potentials, Fermi level and state count from an SCF result archive.

    Lets a band structure be recomputed (another path, solver settings, ...) without
    repeating the SCF.  The archive must come from the same input: the grid size and
    atom positions are checked.
    """
    archive = np.load(archive_path)
    files = set(archive.files)
    if "atom_coordinates_bohr" in files:
        stored = np.asarray(archive["atom_coordinates_bohr"], dtype=float)
        current = np.asarray([atom.position for atom in system.atoms], dtype=float)
        if stored.shape != current.shape or not np.allclose(stored, current, atol=1e-6):
            raise ValueError(f"{archive_path} was written for different atom positions")
    if mode == "scalar":
        keys = {"effective_potential": "output_effective_potential_ry"}
    else:
        keys = {
            name: f"scf_potential_{name}" for name in _REQUIRED[mode]
        }
    missing = [key for key in keys.values() if key not in files]
    if missing:
        raise ValueError(
            f"{archive_path} lacks {missing}; rerun the SCF with this version "
            f"to store the converged potentials needed for mode {mode!r}"
        )
    potentials = {name: np.asarray(archive[key]) for name, key in keys.items()}
    for name, value in potentials.items():
        if value.shape[-1] != system.grid.size:
            raise ValueError(
                f"{archive_path}: potential {name!r} has {value.shape[-1]} grid points, "
                f"the input's grid has {system.grid.size}"
            )
    eigenvalue_key = "eigenvalues_up_ry" if mode == "spin_polarized" else "eigenvalues_ry"
    k_count = int(archive["n_kpoints"]) if "n_kpoints" in files else 1
    number_of_states = int(np.asarray(archive[eigenvalue_key]).size // k_count)
    return potentials, float(archive["fermi_level_ry"]), number_of_states


def spin_expectation(spinors: np.ndarray) -> np.ndarray:
    """``<sigma_x>, <sigma_y>, <sigma_z>`` of every spinor column, shape ``(n, 3)``."""
    n_grid = spinors.shape[0] // 2
    up, down = spinors[:n_grid], spinors[n_grid:]
    cross = np.sum(np.conj(up) * down, axis=0)
    return np.column_stack(
        (
            2.0 * cross.real,
            2.0 * cross.imag,
            np.sum(np.abs(up) ** 2, axis=0) - np.sum(np.abs(down) ** 2, axis=0),
        )
    )


def run_band_structure(
    system,
    settings: BandStructureSettings,
    *,
    mode: str,
    potentials: dict[str, np.ndarray],
    fermi_level: float,
    number_of_states: int,
    callback: Callable[[int, int, np.ndarray, float, int], None] | None = None,
    residual_tolerance: float = 2.0e-2,
    max_attempts: int = 5,
    workers: int | str | None = None,
    backend: str | None = None,
) -> BandStructureResult:
    """Diagonalize the SCF Hamiltonian along ``settings``' path.

    ``potentials`` are the converged local potentials (:func:`band_potentials`),
    ``number_of_states`` the number of bands reported per k-point (and per
    spin channel).  ``callback(k_index, n_kpoints, eigenvalues, residual,
    attempts)`` is called as each k-point finishes (in completion order when
    several workers run).  ``residual_tolerance`` [Ry] and ``max_attempts``
    control the CHEBDAV re-solve described in the module docstring (a CHEBFF
    first solver is never retried).

    The k-points of the path are independent, so they are spread over
    ``workers`` worker threads (``None``: ``PARSEC_KPOINT_WORKERS``, default
    automatic; see :mod:`parsec_python.Parallel`).  ``backend`` (or ``PARSEC_KPOINT_BACKEND``) selects ``cpu`` (default),
    ``cupy`` (one worker thread per CUDA device; the Hamiltonian and the
    Chebyshev filter run on the GPU), ``auto`` or ``numpy-device`` (the device
    code on the host).
    Each k-point is the same computation whatever the worker count, so the
    bands do not depend on it.
    """

    if mode not in MODES:
        raise ValueError(f"unknown band mode {mode!r}; choose from {MODES}")
    missing = [name for name in _REQUIRED[mode] if name not in potentials]
    if missing:
        raise ValueError(f"mode {mode!r} needs the potentials {missing}")
    cell = system.input.periodic_cell
    segments, names = band_path_cartesian(settings, cell)
    k_points = np.concatenate(segments)
    counts = np.asarray([len(segment) for segment in segments])

    eigval_settings = _eigval_settings(system, system.input.eigensolver.filter_degree)
    spinor = mode in ("soc", "soc_spin_polarized", "noncollinear")
    channels = ("up", "down") if mode == "spin_polarized" else ("all",)
    n_k = k_points.shape[0]
    eigenvalues = np.zeros((len(channels), n_k, number_of_states))
    residuals = np.zeros((len(channels), n_k))
    attempts = np.ones((len(channels), n_k), dtype=int)
    sigma = np.zeros((1, n_k, number_of_states, 3)) if spinor else None

    def hamiltonians(k_point):
        """One ``(operator, is_spinor)`` per channel at ``k_point``."""
        nonlocal_k, soc_k = _k_point_operators(system, k_point[None, :])[0]
        if mode == "scalar":
            return [
                KPointKohnShamHamiltonian(
                    system.negative_laplacian,
                    system.gradient,
                    potentials["effective_potential"],
                    nonlocal_k,
                    k_point,
                )
            ]
        if mode == "spin_polarized":
            return [
                KPointKohnShamHamiltonian(
                    system.negative_laplacian,
                    system.gradient,
                    potentials[name],
                    nonlocal_k,
                    k_point,
                )
                for name in ("potential_up", "potential_down")
            ]
        extra = {}
        if mode == "soc_spin_polarized":
            extra["xc_delta"] = potentials["xc_delta"]
        elif mode == "noncollinear":
            extra["xc_field"] = potentials["xc_field"]
        return [
            _spinor_hamiltonian(
                system,
                k_point,
                nonlocal_k,
                soc_k,
                potentials["effective_potential"],
                **extra,
            )
        ]

    def solve(operator, attempt: int):
        """One diagonalization; later attempts use a larger CHEBDAV block and a new seed."""
        settings_now = eigval_settings
        if attempt and eigval_settings.initial_method == "chebdav":
            base = eigval_settings.chebdav
            settings_now = dataclasses.replace(
                eigval_settings,
                chebdav=dataclasses.replace(
                    base,
                    block_size=base.block_size + 2 * attempt,
                    random_seed=base.random_seed + attempt,
                ),
            )
        solution = solve_eigval(operator, number_of_states, settings=settings_now, state=None)
        values = np.asarray(solution.eigenvalues, dtype=float)
        vectors = np.asarray(solution.vectors)
        residual = np.asarray(operator @ vectors) - vectors * values[None, :]
        return values, vectors, float(np.max(np.linalg.norm(residual, axis=0)))

    can_retry = eigval_settings.initial_method == "chebdav"

    def solve_point(k_point, converter=None):
        """Bands of one k-point (all channels) with the CHEBDAV re-solve."""
        point_values = np.zeros((len(channels), number_of_states))
        point_residuals = np.zeros(len(channels))
        point_attempts = np.ones(len(channels), dtype=int)
        point_sigma = np.zeros((number_of_states, 3)) if spinor else None
        for channel, hamiltonian in enumerate(hamiltonians(k_point)):
            operator = (
                converter.operator(hamiltonian).as_linear_operator()
                if converter is not None
                else hamiltonian.as_linear_operator()
            )
            best = None
            failure = None
            for attempt in range(max_attempts if can_retry else 1):
                try:
                    trial = solve(operator, attempt)
                except FloatingPointError as error:
                    # A diverged Chebyshev recurrence (a bad spectral bound from an
                    # unlucky start vector): try again with another block size and seed.
                    failure = error
                    point_attempts[channel] = attempt + 1
                    continue
                point_attempts[channel] = attempt + 1
                if best is None or trial[2] < best[2]:
                    best = trial
                if trial[2] <= residual_tolerance:
                    break
            if best is None:
                raise FloatingPointError(
                    f"band k-point at {np.round(k_point, 6).tolist()}: every one of "
                    f"{point_attempts[channel]} diagonalization attempts diverged ({failure})"
                ) from failure
            values, vectors, point_residuals[channel] = best
            point_values[channel] = values
            if spinor:
                point_sigma = spin_expectation(vectors)
        return point_values, point_residuals, point_attempts, point_sigma

    from .kpoint_parallel import device_worker_count, resolve_backend

    plan = resolve_backend(backend)

    class _BandWorker:
        def __init__(self, tasks, index):
            self.tasks = tasks
            # Built in the worker (thread) so the device is current for it.
            self.converter = plan.converter(index) if plan is not None else None

        def solve_k(self):
            for k_index in self.tasks:
                yield k_index, solve_point(k_points[k_index], self.converter)

    if plan is not None:
        workers = device_worker_count(plan, n_k)
    with make_pool(
        range(n_k),
        _BandWorker,
        workers=workers,
        problem_size=system.grid.size,
    ) as pool:
        for k_index, (values, point_residuals, point_attempts, point_sigma) in pool.stream(
            "solve_k"
        ):
            eigenvalues[:, k_index] = values
            residuals[:, k_index] = point_residuals
            attempts[:, k_index] = point_attempts
            if spinor:
                sigma[0, k_index] = point_sigma
            if callback is not None:
                callback(
                    k_index,
                    n_k,
                    values,
                    float(point_residuals.max()),
                    int(point_attempts.max()),
                )

    return BandStructureResult(
        mode=mode,
        channels=channels,
        segment_names=names,
        segment_counts=counts,
        k_points=k_points,
        eigenvalues=eigenvalues,
        residual_norms=residuals,
        fermi_level=float(fermi_level),
        spin_expectation=sigma,
        attempts=attempts,
    )


__all__ = [
    "MODES",
    "band_mode",
    "band_path_cartesian",
    "band_potentials",
    "load_scf_for_bands",
    "run_band_structure",
    "spin_expectation",
]
