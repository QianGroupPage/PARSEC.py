"""Fock exchange of a periodic cell: pair potentials by FFT, Bloch orbitals and ACE per k-point.

Conventions are those of :mod:`parsec_python.V_xc.exchange_operator`: the
columns of an orbital block are ``q = sqrt(dV) psi`` (``sum |q|**2 = 1``),
energies and potentials are in Rydberg, occupations ``f`` lie in ``[0, 1]``
per spin and a set of orbitals carries the spin weight ``s`` (2 for an
unpolarized set, 1 for one channel of a spin-polarized run).

The box
-------
A periodic grid is an ``n0 x n1 x n2`` box of points along its step vectors
(:class:`GridBox`): Andrew Grieder's grids store it row-major (a reshape),
the Fortran-derived grid of a slab keeps the active points of its index box
(``to_box``).  The box lattice is ``n_i s_i`` (bulk: the cell; slab: ``a1,
a2`` and the height ``n3 h_z`` of the box along z), the reciprocal vectors
of the FFT indices are those of the box lattice.

Pair potentials
---------------
For a Bloch factor ``u_nk`` (``psi_nk = e^{ik.r} u_nk``) and an occupied one
``u_mk'``, the pair density ``conj(psi_mk') psi_nk`` is a Bloch function of
wave vector ``q = k - k'`` with the periodic part ``rho = conj(u_mk') u_nk /
dV`` (e/bohr**3).  Its potential under the exchange kernel is the Bloch
function of the same ``q`` with the periodic part

``w = ifftn(K(q + G) fftn(rho))``                                      (Ry)

with numpy's FFT conventions (``fftn`` unnormalized, ``ifftn`` with ``1/N``):
``fftn(rho)/N`` are the Fourier coefficients of ``rho`` and ``K(q + G)``
(Ry bohr**2, ``8 pi/|q + G|**2`` for the Coulomb kernel ``2/r``) is the
Fourier transform of the kernel, so no further ``1/Omega`` appears; this is
the convention of :func:`parsec_python.Hartree.gspace.solve_hartree_g_space`.
``K`` comes from :class:`parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel`
(the ``q + G = 0`` element regularized by the treatment of the Coulomb
singularity, ``Exx_Divergence``; a slab's kernel truncated along z,
``Exx_Slab_Coulomb``).  A kernel whose ``shape`` exceeds the box along a
non-periodic axis (the truncated slab kernel on its padded box) gets ``rho``
padded with zeros at the end of that axis and ``w`` cropped.

Exchange operator and energy
----------------------------
With the full Monkhorst-Pack mesh of ``N`` points (weights ``1/N``),

``(V_x u)_nk = - (1/N) sum_k' sum_m f_mk' u_mk' w[conj(u_mk') u_nk; k - k']``,

and ``E_x = (s/2) sum_k w_k sum_n f_nk Re <u_nk | V_x u_nk>`` with the
weights ``w_k`` of the k-points solved.  The treatment of the Coulomb
singularity lives entirely in the ``q + G = 0`` element of ``K`` (it
multiplies the ``G = 0`` coefficient of a pair density, the overlap of the
two factors): there is no separate energy constant.  Those are the representatives of the
mesh under time reversal (:class:`BlochOrbitalSet`): the finite-difference
Bloch operator satisfies ``H_{-k} = conj(H_k)`` exactly, so the factor at
``-k`` is ``conj(u_k)``, used with the label ``-k`` (never moved by a
reciprocal lattice vector).  At the Gamma point alone the orbitals are real
and :class:`PeriodicPairPoisson` serves the exchange operator of
:mod:`parsec_python.V_xc.exchange_operator` unchanged (``q = 0``).

ACE per k-point
---------------
``W = V_x U`` of the orbital block ``U`` of a k-point: ``M = -U^H W = L L^H``
(Hermitian, symmetrized), ``xi = W L^-H``, ``V_ACE = -xi xi^H``
(:class:`BlochCompressedExchange`), the form of Lin, J. Chem. Theory Comput.
12, 2242 (2016) for complex orbitals.

Numerical modules do not print or open files: everything is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy import fft as scipy_fft
from scipy.linalg import solve_triangular

from .exact_exchange import OCCUPATION_THRESHOLD


#: Pair densities solved together unless a caller chooses (``Exx_Batch_Size``).
DEFAULT_BATCH_SIZE = 32
#: Two k-points closer than this (bohr^-1) are one point (time-reversal partners, q keys).
K_TOLERANCE = 1.0e-8


def fft_workers() -> int:
    """Threads of the batched FFTs of the pair potentials: ``PARSEC_FFT_WORKERS``, else half the CPUs of the process.

    Half, as the k-point workers take (``acceleration/Parallel/pool.py``): the
    transforms of large batches are bound by memory bandwidth.  The exchange
    runs between the SCF loops of the Fock loop, when no other stage holds the
    cores; the transforms (scipy's pocketfft) are those of numpy, split over
    the batch.
    """

    value = os.environ.get("PARSEC_FFT_WORKERS", "").strip()
    if value:
        return max(1, int(value))
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if callable(affinity) else (os.cpu_count() or 1)
    return max(1, available // 2)


# ---------------------------------------------------------------------------
# The box of a periodic grid.
# ---------------------------------------------------------------------------


class GridBox:
    """The FFT box of a periodic grid and the map between grid vectors and the box.

    ``shape`` is ``points_per_axis``; ``lattice`` the box lattice (rows ``n_i
    s_i``, bohr); ``periodic_axes`` the axes along which the grid wraps (a
    slab's third axis does not).  ``to_box``/``from_box`` take a batch of grid
    vectors ``(points, nb)`` to ``(nb, n0, n1, n2)`` and back.
    """

    def __init__(self, grid) -> None:
        self.shape = tuple(int(value) for value in grid.points_per_axis)
        self.size = int(grid.size)
        self.volume_element = float(grid.volume_element)
        index_min = getattr(grid, "index_min", None)
        count = int(np.prod(self.shape))
        if index_min is not None:
            # The Fortran-derived wire/slab grid: the active points of its index box.
            local = np.asarray(grid.integer_coordinates, dtype=np.int64) - np.asarray(index_min, dtype=np.int64)
            self.flat = np.ravel_multi_index(tuple(local.T), self.shape)
        elif self.size == count:
            self.flat = None
        else:
            raise NotImplementedError("the exchange of a periodic cell needs a grid that fills its box")
        modes = getattr(grid, "box_boundary_modes", None) or grid.boundary_modes
        self.periodic_axes = tuple(axis for axis, mode in enumerate(modes) if mode == "wrap")
        self.lattice = np.asarray(grid.step_vectors, dtype=np.float64) * np.asarray(self.shape, dtype=np.float64)[:, None]
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(self.lattice).T

    def to_box(self, vectors: np.ndarray) -> np.ndarray:
        """``(points, nb)`` grid vectors as ``(nb, n0, n1, n2)``."""

        vectors = np.asarray(vectors)
        batch = vectors.shape[1]
        if self.flat is None:
            return np.ascontiguousarray(vectors.T).reshape((batch,) + self.shape)
        box = np.zeros((batch, int(np.prod(self.shape))), dtype=vectors.dtype)
        box[:, self.flat] = vectors.T
        return box.reshape((batch,) + self.shape)

    def from_box(self, box: np.ndarray) -> np.ndarray:
        """``(nb, n0, n1, n2)`` as grid vectors ``(points, nb)``."""

        batch = box.shape[0]
        flat = box.reshape(batch, -1)
        if self.flat is not None:
            flat = flat[:, self.flat]
        return np.ascontiguousarray(flat.T)


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------


def kernel_coefficients(functional, fraction: float) -> tuple[float, float, float | None]:
    """``(alpha, beta, omega)`` of the kernel ``alpha/r + beta erfc(omega r)/r`` of a Fock loop.

    The operator of the loop is ``fraction`` times the exchange of this
    kernel (:class:`parsec_python.SCF.hybrid.HybridDefinition`): one kernel
    of the functional gives ``(1, 0)``, ``(0, 1)`` or ``(1, -1)`` (Coulomb,
    erfc, erf) with ``fraction`` its weight, two kernels (CAM-B3LYP) the
    weighted kernel with ``fraction`` 1 -- the CAM coefficients of libxc
    divided by ``fraction``.
    """

    alpha = float(functional.alpha) / float(fraction)
    beta = float(functional.beta) / float(fraction)
    omega = None if beta == 0.0 else float(functional.omega)
    return alpha, beta, omega


#: ``Exx_Divergence`` -> the ``divergence`` and ``x_gamma_extrapolation`` of the kernel (``None``: its default).
DIVERGENCE_CHOICES = {
    None: (None, None),
    "auto": (None, None),
    "gygi-baldereschi": ("gygi-baldereschi", False),
    "gygi-baldereschi-x": ("gygi-baldereschi", True),
    "none": ("none", False),
}


def periodic_exchange_kernel(
    box: GridBox,
    k_mesh,
    *,
    alpha: float,
    beta: float,
    omega,
    divergence=None,
    auxiliary_exponent=None,
    slab_coulomb: str = "padded",
):
    """The exchange kernel of the run (:class:`parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel`).

    ``k_mesh`` are the counts of the full Monkhorst-Pack mesh (``(1, 1, 1)`` at
    the Gamma point; a slab's third count is 1).  ``divergence`` is
    ``Exx_Divergence``: ``auto`` (``None``, the kernel's default: Gygi-Baldereschi
    with the extrapolation on the ``1/r`` component of a bulk cell),
    ``gygi-baldereschi`` (Quantum ESPRESSO with ``x_gamma_extrapolation =
    .false.``), ``gygi-baldereschi-x`` (Quantum ESPRESSO's default) or ``none``;
    ``auxiliary_exponent`` (``Exx_Auxiliary_Exponent``, bohr^2) that of the
    Gygi-Baldereschi auxiliary function (``None``: the kernel's).  A slab's
    ``slab_coulomb`` (``Exx_Slab_Coulomb``): ``padded`` (the bulk kernel on the box
    padded to twice its height), ``truncated`` (the Coulomb kernel cut along z) or
    ``periodic`` (the bulk kernel on the unpadded slab box, Quantum ESPRESSO's convention).
    """

    from ..Hartree.exchange_kernel import PeriodicExchangeKernel

    if divergence not in DIVERGENCE_CHOICES:
        raise ValueError(f"unknown treatment of the Coulomb singularity {divergence!r}")
    treatment, extrapolation = DIVERGENCE_CHOICES[divergence]
    slab = tuple(box.periodic_axes) == (0, 1)
    keywords = dict(
        alpha=alpha,
        beta=beta,
        omega=omega,
        periodic_axes=(0, 1, 2) if slab and slab_coulomb == "periodic" else tuple(box.periodic_axes),
    )
    if slab and slab_coulomb != "periodic":
        keywords["slab_coulomb"] = slab_coulomb
    if treatment is not None:
        keywords["divergence"] = treatment
        keywords["x_gamma_extrapolation"] = extrapolation
    if auxiliary_exponent is not None:
        keywords["auxiliary_exponent"] = float(auxiliary_exponent)
    return PeriodicExchangeKernel(box.lattice, box.shape, tuple(int(n) for n in k_mesh), **keywords)


class PeriodicPairPotentials:
    """``w = ifftn(K(q + G) fftn(rho))`` of batches of pair densities on the box of a periodic grid.

    ``solve(densities, q)`` takes complex grid vectors ``(points, nb)`` of one
    Bloch wave vector ``q`` (bohr^-1) and returns the periodic parts of their
    potentials (Ry).  ``solve_real`` serves real densities at ``q = 0``.
    The kernel values of the q-points met are kept (one array per q).
    ``solves`` and ``seconds`` count the work.
    """

    def __init__(self, box: GridBox, kernel) -> None:
        self.box = box
        self.kernel = kernel
        self.fft_shape = tuple(int(value) for value in getattr(kernel, "shape", box.shape))
        if len(self.fft_shape) != 3 or any(f < b for f, b in zip(self.fft_shape, box.shape)):
            raise ValueError("the kernel's FFT box must hold the grid box")
        for axis, (f, b) in enumerate(zip(self.fft_shape, box.shape)):
            if f != b and axis in box.periodic_axes:
                raise ValueError("a kernel may pad the box along non-periodic axes only")
        self.padded = self.fft_shape != box.shape
        self._values: dict = {}
        self.solves = 0
        self.seconds = 0.0
        describe = getattr(kernel, "describe", None)
        self.description = f"FFT pair potentials on the {'x'.join(str(n) for n in self.fft_shape)} box, " + (
            describe() if callable(describe) else str(getattr(kernel, "description", type(kernel).__name__))
        )

    def kernel_values(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        key = tuple(np.round(q / K_TOLERANCE).astype(np.int64))
        values = self._values.get(key)
        if values is None:
            values = np.asarray(self.kernel.values(q), dtype=np.float64)
            if values.shape != self.fft_shape:
                raise ValueError("the kernel values do not match its FFT box")
            self._values[key] = values
        return values

    def forget(self) -> None:
        """Drop the kept kernel values (one exchange evaluation keeps those it met)."""

        self._values.clear()

    def _padded(self, box: np.ndarray) -> np.ndarray:
        if not self.padded:
            return box
        padded = np.zeros((box.shape[0],) + self.fft_shape, dtype=box.dtype)
        padded[(slice(None),) + tuple(slice(0, n) for n in self.box.shape)] = box
        return padded

    def _cropped(self, box: np.ndarray) -> np.ndarray:
        if not self.padded:
            return box
        return box[(slice(None),) + tuple(slice(0, n) for n in self.box.shape)]

    def solve(self, densities: np.ndarray, q) -> np.ndarray:
        """Potentials of complex pair densities ``(points, nb)`` of Bloch wave vector ``q``."""

        started = perf_counter()
        kernel = self.kernel_values(q)
        box = self._padded(self.box.to_box(np.asarray(densities, dtype=np.complex128)))
        workers = fft_workers()
        transformed = scipy_fft.fftn(box, axes=(1, 2, 3), workers=workers, overwrite_x=True)
        transformed *= kernel[None]
        solved = scipy_fft.ifftn(transformed, axes=(1, 2, 3), workers=workers, overwrite_x=True)
        potentials = self.box.from_box(self._cropped(solved))
        self.solves += densities.shape[1]
        self.seconds += perf_counter() - started
        return potentials

    def solve_real(self, densities: np.ndarray) -> np.ndarray:
        """Potentials of real pair densities ``(points, nb)`` at ``q = 0`` (real: ``K(G) = K(-G)``)."""

        started = perf_counter()
        kernel = self.kernel_values(np.zeros(3))
        half = kernel[..., : self.fft_shape[2] // 2 + 1]
        box = self._padded(self.box.to_box(np.asarray(densities, dtype=np.float64)))
        workers = fft_workers()
        transformed = scipy_fft.rfftn(box, axes=(1, 2, 3), workers=workers)
        transformed *= half[None]
        solved = scipy_fft.irfftn(transformed, s=self.fft_shape, axes=(1, 2, 3), workers=workers, overwrite_x=True)
        potentials = self.box.from_box(self._cropped(solved))
        self.solves += densities.shape[1]
        self.seconds += perf_counter() - started
        return potentials


class PeriodicPairPoisson:
    """The pair solver of a periodic cell at the Gamma point (real orbitals).

    The :class:`parsec_python.V_xc.exchange_operator.PairPoissonSolver` of the
    Fock loop of :mod:`parsec_python.SCF.hybrid`: ``solve_batch(rho)`` returns
    ``ifftn(K(G) fftn(rho))`` of real pair densities ``(points, nb)``, real
    because the kernel is even in ``G``.
    """

    def __init__(self, potentials: PeriodicPairPotentials) -> None:
        self.potentials = potentials
        self.description = potentials.description

    @property
    def solves(self) -> int:
        return self.potentials.solves

    @property
    def seconds(self) -> float:
        return self.potentials.seconds

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:
        densities = np.asarray(densities, dtype=np.float64)
        if densities.ndim != 2 or densities.shape[0] != self.potentials.box.size:
            raise ValueError("pair densities must have shape (grid points, batch)")
        return self.potentials.solve_real(densities)


def periodic_pair_solver(
    grid,
    functional,
    fraction: float,
    *,
    k_mesh=(1, 1, 1),
    divergence=None,
    auxiliary_exponent=None,
    slab_coulomb: str = "padded",
) -> PeriodicPairPotentials:
    """:class:`PeriodicPairPotentials` of a hybrid's kernel on ``grid`` (:func:`kernel_coefficients`).

    ``divergence``, ``auxiliary_exponent`` and ``slab_coulomb``: :func:`periodic_exchange_kernel`.
    """

    box = GridBox(grid)
    alpha, beta, omega = kernel_coefficients(functional, fraction)
    kernel = periodic_exchange_kernel(
        box,
        tuple(int(n) for n in k_mesh),
        alpha=alpha,
        beta=beta,
        omega=omega,
        divergence=divergence,
        auxiliary_exponent=auxiliary_exponent,
        slab_coulomb=slab_coulomb,
    )
    return PeriodicPairPotentials(box, kernel)


# ---------------------------------------------------------------------------
# Bloch orbitals and their exchange.
# ---------------------------------------------------------------------------


def time_reversal_partners(k_points, box_reciprocal) -> np.ndarray:
    """Whether ``-k`` is a point of the mesh other than ``k`` (``2k`` not a reciprocal lattice vector)."""

    k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    fractions = 2.0 * k_points @ np.linalg.inv(np.asarray(box_reciprocal, dtype=np.float64))
    return ~np.all(np.abs(fractions - np.round(fractions)) < 1.0e-8, axis=1)


@dataclass(frozen=True)
class BlochOrbitalSet:
    """The Bloch factors of one spin channel (or of both, ``spin_weight`` 2) at the k-points solved.

    ``k_points`` (Cartesian bohr^-1) are the representatives of a uniform
    mesh of ``mesh_size`` points under time reversal, ``weights`` their
    weights (``2/N``, or ``1/N`` where ``partners`` is false: ``-k = k``
    modulo a reciprocal lattice vector).  ``orbitals[t]`` is the ``(points,
    states)`` complex block of k-point ``t`` and ``occupations[t]`` its
    occupations (``[0, 1]`` per spin).
    """

    k_points: np.ndarray
    weights: np.ndarray
    orbitals: tuple
    occupations: tuple
    spin_weight: float
    partners: np.ndarray
    mesh_size: int

    def sources(self, occupation_threshold: float = OCCUPATION_THRESHOLD):
        """``(k', occupied columns of u_k', their occupations)`` of every point of the full mesh."""

        sources = []
        for t, (k_point, block, values) in enumerate(zip(self.k_points, self.orbitals, self.occupations)):
            occupied = np.flatnonzero(np.asarray(values) > occupation_threshold)
            if not occupied.size:
                continue
            columns = np.ascontiguousarray(block[:, occupied])
            weights = np.asarray(values, dtype=np.float64)[occupied]
            sources.append((np.asarray(k_point, dtype=np.float64), columns, weights))
            if self.partners[t]:
                sources.append((-np.asarray(k_point, dtype=np.float64), np.conj(columns), weights))
        return sources


def bloch_orbital_set(k_points, weights, orbitals, occupations, spin_weight: float, box: GridBox) -> BlochOrbitalSet:
    """A :class:`BlochOrbitalSet`, checked to be a time-reversal-reduced uniform mesh."""

    k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64)
    partners = time_reversal_partners(k_points, box.reciprocal)
    mesh_size = int(np.sum(np.where(partners, 2, 1)))
    expected = np.where(partners, 2.0, 1.0) / float(mesh_size)
    # No two k-points solved may be one point or each other's partner (k_i = +-k_j modulo G).
    fractions = k_points @ np.linalg.inv(np.asarray(box.reciprocal, dtype=np.float64))
    upper = np.triu(np.ones((k_points.shape[0], k_points.shape[0]), dtype=bool), 1)
    doubled = False
    for sign in (1.0, -1.0):
        offsets = fractions[:, None, :] + sign * fractions[None, :, :]
        same = np.all(np.abs(offsets - np.round(offsets)) < 1.0e-8, axis=2)
        doubled = doubled or bool(np.any(same & upper))
    if doubled or weights.shape != (k_points.shape[0],) or not np.allclose(weights, expected, rtol=0.0, atol=1.0e-10):
        raise NotImplementedError(
            "the exchange of a periodic cell needs a uniform Monkhorst-Pack mesh reduced by time reversal only "
            "(weights 2/N, or 1/N for k = -k)"
        )
    blocks = tuple(np.asarray(block, dtype=np.complex128) for block in orbitals)
    values = tuple(np.asarray(item, dtype=np.float64) for item in occupations)
    if len(blocks) != k_points.shape[0] or len(values) != k_points.shape[0]:
        raise ValueError("one orbital block and one set of occupations per k-point are needed")
    for block, item in zip(blocks, values):
        if block.ndim != 2 or block.shape[0] != box.size or item.shape != (block.shape[1],):
            raise ValueError("the orbital blocks must have shape (grid points, states) with one occupation per state")
    return BlochOrbitalSet(k_points, weights, blocks, values, float(spin_weight), partners, mesh_size)


@dataclass(frozen=True)
class BlochExchangeApplication:
    """``W_t = V_x U_t`` of every k-point of an orbital set and what it took.

    ``applied[t]`` has the shape of ``orbitals[t]``; its columns not in
    ``columns[t]`` are zero.  ``energy`` is ``E_x`` (Ry, unscaled, spin
    weight included) of the occupied orbitals,
    ``nan`` where not every occupied column was applied.  ``occupied`` holds
    the occupied columns of every k-point (empty for a set without
    electrons).
    """

    applied: tuple
    columns: tuple
    occupied: tuple
    energy: float
    pair_solves: int
    poisson_seconds: float
    seconds: float

    @property
    def has_electrons(self) -> bool:
        return any(len(columns) for columns in self.occupied)


def apply_bloch_exchange(
    orbital_set: BlochOrbitalSet,
    potentials: PeriodicPairPotentials,
    *,
    columns: str = "all",
    known: BlochExchangeApplication | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    occupation_threshold: float = OCCUPATION_THRESHOLD,
    pair_symmetry: bool = True,
) -> BlochExchangeApplication:
    """``W_t = V_x U_t`` for every k-point ``t`` of ``orbital_set`` (see the module).

    ``columns``: ``"all"`` or ``"occupied"`` (enough for ``E_x``).  ``known``
    is an earlier application to the same orbitals (of the occupied columns):
    its columns are kept and only the others are solved.  Loop order: k-point
    ``t`` of the targets, k-point ``k'`` of the full mesh, batches of
    ``batch_size`` pairs (target column, occupied column of ``k'``).

    ``pair_symmetry``: a pair of two occupied states that are both targets is
    solved once for both.  Target ``(t, n)`` with the source ``(r, m)`` at
    ``+k_r`` and target ``(r, m)`` with the source ``(t, n)`` at ``+k_t`` have
    conjugate densities and opposite ``q``: the second potential is
    ``conj(w)`` (exact: ``K(-q - G) = K(q + G)`` element by element, the
    Nyquist planes of the kernel averaged).  With the source ``(r, m)`` at
    ``-k_r`` (``conj(u)``) and the source ``(t, n)`` at ``-k_t`` the density
    ``u_rm u_tn`` and ``q = k_t + k_r`` are the same: the potential is ``w``.
    ``False`` solves every pair (the reference of the check).
    """

    started = perf_counter()
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if columns not in ("all", "occupied"):
        raise ValueError("columns must be 'all' or 'occupied'")
    box = potentials.box
    inverse_volume = 1.0 / box.volume_element
    scale = 1.0 / float(orbital_set.mesh_size)
    count = len(orbital_set.orbitals)
    occupied_columns = []
    results, done_sets, target_lists = [], [], []
    for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
        occupied = [int(index) for index in np.flatnonzero(values > occupation_threshold)]
        occupied_columns.append(tuple(occupied))
        wanted = list(range(block.shape[1])) if columns == "all" else list(occupied)
        if known is not None:
            if known.applied[t].shape != block.shape or tuple(known.occupied[t]) != tuple(occupied):
                raise ValueError("the known application belongs to another orbital set")
            results.append(known.applied[t].copy())
            done = set(known.columns[t])
        else:
            results.append(np.zeros(block.shape, dtype=np.complex128))
            done = set()
        done_sets.append(done)
        target_lists.append([column for column in wanted if column not in done])
    target_sets = [set(items) for items in target_lists]
    occupied_sets = [set(items) for items in occupied_columns]
    # The points of the full mesh with occupied states: (rep, sign), +k_r with u, -k_r (a partner) with conj(u).
    sources = []
    for r in range(count):
        if occupied_columns[r]:
            sources.append((r, 1))
            if orbital_set.partners[r]:
                sources.append((r, -1))
    covered: set = set()
    pair_solves = 0
    poisson_seconds = 0.0
    potentials.forget()
    for t in range(count):
        targets = target_lists[t]
        if not targets or not sources:
            continue
        block = orbital_set.orbitals[t]
        k_point = np.asarray(orbital_set.k_points[t], dtype=np.float64)
        for r, sign in sources:
            columns_r = list(occupied_columns[r])
            source_block = orbital_set.orbitals[r][:, columns_r]
            if sign < 0:
                source_block = np.conj(source_block)
            source_occupations = orbital_set.occupations[r][columns_r]
            q = k_point - sign * np.asarray(orbital_set.k_points[r], dtype=np.float64)
            # (target column n, source position j, mirrored pair or None)
            pairs = []
            for n in targets:
                for j, m in enumerate(columns_r):
                    if (t, n, r, sign, m) in covered:
                        continue
                    mirror = None
                    if (
                        pair_symmetry
                        and n in occupied_sets[t]
                        and m in target_sets[r]
                        and not (r == t and m == n)
                        and (sign > 0 or orbital_set.partners[t])
                    ):
                        mirror = (r, m, t, sign, n)
                        covered.add(mirror)
                    pairs.append((n, j, mirror))
            for begin in range(0, len(pairs), batch_size):
                batch = pairs[begin : begin + batch_size]
                first = np.fromiter((pair[0] for pair in batch), dtype=np.int64, count=len(batch))
                second = np.fromiter((pair[1] for pair in batch), dtype=np.int64, count=len(batch))
                densities = np.conj(source_block[:, second]) * block[:, first]
                densities *= inverse_volume
                solve_started = perf_counter()
                solved = potentials.solve(densities, q)
                poisson_seconds += perf_counter() - solve_started
                contributions = source_block[:, second] * solved
                contributions *= (-scale * source_occupations[second])[None, :]
                for index, (n, _, mirror) in enumerate(batch):
                    results[t][:, n] += contributions[:, index]
                    if mirror is None:
                        continue
                    # W_r[:, m] -= (1/N) f_tn s_tn w', s_tn = u_tn (w' = conj(w)) or conj(u_tn) (w' = w)
                    _, m, _, _, _ = mirror
                    if sign > 0:
                        term = block[:, n] * np.conj(solved[:, index])
                    else:
                        term = np.conj(block[:, n]) * solved[:, index]
                    results[r][:, m] -= (scale * orbital_set.occupations[t][n]) * term
                pair_solves += len(batch)
    applied = results
    done_columns = [tuple(sorted(done | set(targets))) for done, targets in zip(done_sets, target_lists)]
    complete = all(set(occupied) <= set(done) for occupied, done in zip(occupied_columns, done_columns))
    energy = float("nan")
    if complete:
        energy = 0.0
        for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
            for column in occupied_columns[t]:
                energy += (
                    orbital_set.weights[t]
                    * values[column]
                    * float(np.vdot(block[:, column], applied[t][:, column]).real)
                )
        energy *= 0.5 * orbital_set.spin_weight
    return BlochExchangeApplication(
        applied=tuple(applied),
        columns=tuple(done_columns),
        occupied=tuple(occupied_columns),
        energy=float(energy),
        pair_solves=pair_solves + (0 if known is None else known.pair_solves),
        poisson_seconds=poisson_seconds + (0.0 if known is None else known.poisson_seconds),
        seconds=perf_counter() - started + (0.0 if known is None else known.seconds),
    )


@dataclass(frozen=True)
class BlochCompressedExchange:
    """``V_ACE = -xi xi^H`` of the orbital block of one k-point (complex ``xi``, ``(points, states)``).

    ``fraction`` scales the operator where it joins a Hamiltonian; the
    methods with ``scaled=False`` give the bare compressed exchange.
    ``asymmetry`` is ``max|M - M^H| / max|M|`` of the block.
    """

    projectors: np.ndarray
    fraction: float = 1.0
    asymmetry: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        size = int(self.projectors.shape[0])
        return (size, size)

    @property
    def rank(self) -> int:
        return int(self.projectors.shape[1])

    def apply(self, vectors: np.ndarray, *, scaled: bool = True) -> np.ndarray:
        """``a_x V_ACE X = -a_x xi (xi^H X)``."""

        xi = self.projectors
        factor = -(self.fraction if scaled else 1.0)
        return factor * (xi @ (xi.conj().T @ vectors))

    def expectation_values(self, vectors: np.ndarray, *, scaled: bool = True) -> np.ndarray:
        """``<x_k|a_x V_ACE|x_k> = -a_x ||xi^H x_k||^2`` for each column (real)."""

        block = vectors[:, None] if np.ndim(vectors) == 1 else vectors
        overlaps = self.projectors.conj().T @ block
        factor = -(self.fraction if scaled else 1.0)
        return factor * np.einsum("ij,ij->j", overlaps.conj(), overlaps).real


def compress_bloch_exchange(orbitals: np.ndarray, applied: np.ndarray, *, fraction: float = 1.0) -> BlochCompressedExchange:
    """``V_ACE`` of one k-point from ``W = V_x U`` (every column applied)."""

    orbitals = np.asarray(orbitals, dtype=np.complex128)
    applied = np.asarray(applied, dtype=np.complex128)
    if applied.shape != orbitals.shape:
        raise ValueError("the exchange application does not match the orbitals")
    overlap = -(orbitals.conj().T @ applied)
    scale = float(np.max(np.abs(overlap), initial=0.0))
    asymmetry = float(np.max(np.abs(overlap - overlap.conj().T), initial=0.0)) / scale if scale > 0.0 else 0.0
    overlap = 0.5 * (overlap + overlap.conj().T)
    try:
        lower = np.linalg.cholesky(overlap)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "the exchange matrix M = -U^H V_x U of a k-point is not positive definite; ACE needs orbitals "
            "that overlap the occupied ones"
        ) from error
    # xi = W L^-H  <=>  xi^H = L^-1 W^H.
    projectors = solve_triangular(lower, applied.conj().T, lower=True).conj().T
    return BlochCompressedExchange(np.ascontiguousarray(projectors), float(fraction), asymmetry)


def bloch_band_energy(operators: Sequence, orbital_set: BlochOrbitalSet) -> float:
    """``E_x(P, P0) = (s/2) sum_t w_t sum_n f_nt <u_nt|V_ACE_t|u_nt>`` (Ry, unscaled) of ``orbital_set``."""

    total = 0.0
    for t, (operator, block, values) in enumerate(zip(operators, orbital_set.orbitals, orbital_set.occupations)):
        if operator is None:
            continue
        occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
        if not occupied.size:
            continue
        expectation = operator.expectation_values(block[:, occupied], scaled=False)
        total += orbital_set.weights[t] * float(np.dot(values[occupied], expectation))
    return 0.5 * orbital_set.spin_weight * total


__all__ = [
    "BlochCompressedExchange",
    "BlochExchangeApplication",
    "BlochOrbitalSet",
    "DEFAULT_BATCH_SIZE",
    "GridBox",
    "PeriodicPairPoisson",
    "PeriodicPairPotentials",
    "apply_bloch_exchange",
    "bloch_band_energy",
    "bloch_orbital_set",
    "compress_bloch_exchange",
    "fft_workers",
    "kernel_coefficients",
    "periodic_exchange_kernel",
    "periodic_pair_solver",
    "time_reversal_partners",
]
