"""Hartree boundary values of a slab (2-D periodic) system (F ``hartset_slab.F90``), Fortran-derived routines.

The module ``Hartree/slab.py`` of the parity round, used by the wire/slab path of the
Fortran-derived routines (:mod:`parsec_python.Hartree.confined`); Andrew Grieder's slab Hartree solver of the
default routines is :mod:`parsec_python.Hartree.slab`.

A slab is periodic along ``a1`` and ``a2`` (in the ``xy`` plane, surface
cell area ``A``) and confined to ``|z| <= R``.  The density of each grid plane
``z_k`` is a periodic sheet; its potential at height ``z`` is (Lennard-Jones
and Dent, Trans. Faraday Soc. 1927; Hartree units)

``V_k(r, z) = -2 pi sigma_k |z - z_k|
              + sum_(G != 0) (2 pi/(A |G|)) e^(-|G| |z - z_k|) n_k(G) e^(i G.r)``,

with ``sigma_k`` the charge per area of the plane and ``n_k(G)`` its
Fourier components.  Summing ``G`` and ``-G`` together over the half plane
``H = {m = 1..L, l = -L..L} U {m = 0, l = 1..L}`` of ``G = l b1 + m b2``
(``L = Solver_Lpole``) gives Fortran's boundary value at an exterior point
``(r_b, z_b)``

``V_b = 2 sum_k [ sigma'_k |z_b - z_k|
        + sum_H 2 h^3 e^(-|G| |z_b - z_k|) (cos(G.r_b) qc_k(G) + sin(G.r_b) qs_k(G)) ]``  (Ry),

``sigma'_k = -2 pi h_3 <n>_k`` (the mean density of the plane times its
spacing), ``qc_k(G) = (2 pi/(A |G|)) sum_(j in k) n_j cos(G.r_j)`` and ``qs``
the same with ``sin`` (``hartset_slab.F90:200-400``).  Only the ``+-z``
neighbours of a slab grid lie outside the domain.  The divergent constant of
the ``G = 0`` term is dropped, as in the slab Ewald sum of the ions
(:func:`parsec_python.V_ion.wire_slab.slab_ewald_energy`).

Deviation from Fortran (``fortran_plane_quirk=True`` reproduces it): the
planes of ``hartset_slab`` run from ``grid%nzmin + norder + 1``, which leaves
out the lowest plane of the index box (``hartset_slab.F90:136``).  That plane
holds grid points when ``int(2 R/h)`` is odd and the grid shift along ``z`` is
one half; its charge is then missing from the boundary values.  (With more
than one MPI rank Fortran also leaves the top plane out of the global sums
of ``qclm``, ``qslm`` and ``sigma``, which is empty in the same cases.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlabBoundaryPlan:
    """Static tables of the slab boundary values of one grid.

    ``vectors`` holds the ``(l, m)`` of the half plane, ``lengths`` the
    ``|G|``; ``plane_cos``/``plane_sin`` (``(G, n1, n2)``) the in-plane
    phases of the grid points and ``boundary_cos``/``boundary_sin``
    (``(B, G)``) those of the exterior points; ``decay`` is
    ``exp(-|G| |k_b - k| h_3)`` for every distinct exterior plane and every
    source plane, ``(G, distinct exterior planes, n3)``, and ``distance`` the
    plane distances ``|k_b - k| h_3``.
    """

    order: int
    area: float
    volume_element: float
    plane_spacing: float
    vectors: np.ndarray
    lengths: np.ndarray
    plane_cos: np.ndarray
    plane_sin: np.ndarray
    plane_used: np.ndarray  # (n3,) 1.0 for a plane that enters the sums
    plane_counts: np.ndarray  # (n3,) points of each plane of the index box
    boundary_plane: np.ndarray  # (B,) index into the distinct exterior planes
    boundary_cos: np.ndarray
    boundary_sin: np.ndarray
    decay: np.ndarray
    distance: np.ndarray  # (distinct exterior planes, n3)
    to_box: object

    def plane_moments(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(qc, qs, sigma')`` of ``density``: ``(G, n3)``, ``(G, n3)``, ``(n3,)``."""

        box = self.to_box(np.asarray(density, dtype=np.float64))
        prefactor = 2.0 * np.pi / (self.area * self.lengths)
        qc = np.einsum("gij,ijk->gk", self.plane_cos, box) * prefactor[:, None]
        qs = np.einsum("gij,ijk->gk", self.plane_sin, box) * prefactor[:, None]
        sums = box.sum(axis=(0, 1))
        sigma = np.zeros_like(sums)
        filled = self.plane_counts > 0
        sigma[filled] = -2.0 * np.pi * self.plane_spacing * sums[filled] / self.plane_counts[filled]
        used = self.plane_used
        return qc * used[None, :], qs * used[None, :], sigma * used

    def boundary_values(self, density: np.ndarray) -> np.ndarray:
        """Rydberg potential of ``density`` at every exterior stencil point."""

        qc, qs, sigma = self.plane_moments(density)
        # (G, P, k) x (G, k) -> (G, P)
        cosine_part = np.einsum("gpk,gk->gp", self.decay, qc)
        sine_part = np.einsum("gpk,gk->gp", self.decay, qs)
        plane = self.boundary_plane
        periodic = np.einsum("bg,gb->b", self.boundary_cos, cosine_part[:, plane])
        periodic = periodic + np.einsum("bg,gb->b", self.boundary_sin, sine_part[:, plane])
        mean = self.distance @ sigma
        return 2.0 * (mean[plane] + 2.0 * self.volume_element * periodic)


def slab_reciprocal_half_plane(order: int) -> np.ndarray:
    """The ``(l, m)`` of Fortran's half plane, in its loop order."""

    pairs = [(l, m) for l in range(-order, order + 1) for m in range(1, order + 1)]
    pairs += [(l, 0) for l in range(1, order + 1)]
    return np.asarray(pairs, dtype=np.int64)


def build_slab_boundary_plan(
    grid,
    exterior_integer_points: np.ndarray,
    order: int,
    *,
    fortran_plane_quirk: bool = False,
) -> SlabBoundaryPlan:
    """Tables of :class:`SlabBoundaryPlan` for ``grid`` and its exterior points.

    ``exterior_integer_points`` are the integer coordinates of the missing
    stencil neighbours (all of them ``+-z`` neighbours).
    """

    order = int(order)
    if order < 1:
        raise ValueError("the slab boundary needs Solver_Lpole >= 1")
    lattice = grid.cell.lattice_vectors
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    vectors = slab_reciprocal_half_plane(order)
    cartesian = vectors[:, 0, None] * reciprocal[0][None, :] + vectors[:, 1, None] * reciprocal[1][None, :]
    lengths = np.sqrt(np.sum(cartesian**2, axis=1))
    lengths_axis = np.sqrt(np.sum(lattice[:2] ** 2, axis=1))
    # Fortran's phase: klm . (u, v) with klm = (l 2 pi/|a1|, m 2 pi/|a2|) and
    # u, v the coordinates along the unit lattice vectors.
    wave = 2.0 * np.pi / lengths_axis
    n1, n2, n3 = grid.box_shape
    i = np.arange(grid.index_min[0], grid.index_max[0] + 1)
    j = np.arange(grid.index_min[1], grid.index_max[1] + 1)
    u = (i + grid.shift[0]) * grid.spacings[0]
    v = (j + grid.shift[1]) * grid.spacings[1]
    phase = (
        vectors[:, 0, None, None] * wave[0] * u[None, :, None]
        + vectors[:, 1, None, None] * wave[1] * v[None, None, :]
    )
    plane_cos = np.cos(phase)
    plane_sin = np.sin(phase)

    box_count = np.zeros(grid.box_shape, dtype=np.float64)
    local = grid.integer_coordinates - grid.index_min
    box_count[local[:, 0], local[:, 1], local[:, 2]] = 1.0
    plane_counts = box_count.sum(axis=(0, 1))
    plane_used = np.ones(n3, dtype=np.float64)
    if fortran_plane_quirk:
        plane_used[0] = 0.0

    exterior = np.asarray(exterior_integer_points, dtype=np.int64)
    wrapped = grid.wrap_periodic(exterior)
    eu = (wrapped[:, 0] + grid.shift[0]) * grid.spacings[0]
    ev = (wrapped[:, 1] + grid.shift[1]) * grid.spacings[1]
    boundary_phase = (
        eu[:, None] * vectors[None, :, 0] * wave[0] + ev[:, None] * vectors[None, :, 1] * wave[1]
    )
    distinct, boundary_plane = np.unique(exterior[:, 2], return_inverse=True)
    source_planes = np.arange(grid.index_min[2], grid.index_max[2] + 1)
    separation = np.abs(distinct[:, None] - source_planes[None, :])
    distance = separation.astype(np.float64) * grid.spacings[2]
    decay = np.exp(-lengths[:, None, None] * distance[None, :, :])
    return SlabBoundaryPlan(
        order=order,
        area=grid.surface_area,
        volume_element=grid.volume_element,
        plane_spacing=float(grid.spacings[2]),
        vectors=vectors,
        lengths=lengths,
        plane_cos=plane_cos,
        plane_sin=plane_sin,
        plane_used=plane_used,
        plane_counts=plane_counts,
        boundary_plane=np.asarray(boundary_plane, dtype=np.int64).reshape(-1),
        boundary_cos=np.cos(boundary_phase),
        boundary_sin=np.sin(boundary_phase),
        decay=decay,
        distance=distance,
        to_box=grid.to_box,
    )


__all__ = ["SlabBoundaryPlan", "build_slab_boundary_plan", "slab_reciprocal_half_plane"]
