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

The Hamiltonian follows the SCF mode, so the same ``bandstruc`` block gives
scalar, collinear spin-polarized (up/down), spin-orbit, spin-orbit +
collinear-spin and non-collinear bands.  Comparing runs with and without
``SO_PSP``/``SO_from_scratch`` gives the effect of spin-orbit coupling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..Eigensolvers import solve_eigval
from ..Grid import band_path_kpoints, explicit_kpoints
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..models import BandStructureResult, BandStructureSettings
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
    callback: Callable[[int, int, np.ndarray, float], None] | None = None,
) -> BandStructureResult:
    """Diagonalize the SCF Hamiltonian along ``settings``' path.

    ``potentials`` are the converged local potentials (:func:`band_potentials`),
    ``number_of_states`` the number of bands reported per k-point (and per
    spin channel).  ``callback(k_index, n_kpoints, eigenvalues, residual)`` is
    called after each k-point.
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

    for index, k_point in enumerate(k_points):
        worst = 0.0
        for channel, hamiltonian in enumerate(hamiltonians(k_point)):
            operator = hamiltonian.as_linear_operator()
            solution = solve_eigval(
                operator, number_of_states, settings=eigval_settings, state=None
            )
            values = np.asarray(solution.eigenvalues, dtype=float)
            vectors = np.asarray(solution.vectors)
            residual = np.asarray(operator @ vectors) - vectors * values[None, :]
            residuals[channel, index] = float(np.max(np.linalg.norm(residual, axis=0)))
            eigenvalues[channel, index] = values
            if spinor:
                sigma[0, index] = spin_expectation(vectors)
            worst = max(worst, residuals[channel, index])
        if callback is not None:
            callback(index, n_k, eigenvalues[:, index], worst)

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
    )


__all__ = [
    "MODES",
    "band_mode",
    "band_path_cartesian",
    "band_potentials",
    "run_band_structure",
    "spin_expectation",
]
