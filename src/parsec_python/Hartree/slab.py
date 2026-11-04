"""
Hartree potential of a slab: periodic in the xy plane, open along z.

The charge sits in a box of ``n3`` planes (:class:`~parsec_python.Grid.pbc.SlabRealSpaceGrid`).  Poisson's equation
is solved inside the box by conjugate gradients with the potential at the points just beyond the first and last plane
taken from the exact solution of the open problem there, as the cluster solver does with its multipole expansion
(and the Fortran ``hartset_slab``, which uses the same expansion in sheets of charge):

    V(x, y, z) = -2 pi sum_k sigma_k |z - z_k|
                 + sum_{G != 0} (2 pi hz / |G|) sum_k rho_G(z_k) exp(-|G| |z - z_k|) exp(i G . r)           [Hartree]

with ``sigma_k = hz <rho_k>`` the charge per area of plane ``k`` and ``rho_G`` the in-plane Fourier coefficients of
the density.  The potential keeps the linear growth of a charged sheet (no mean is removed): it is the one whose
ion-ion and ionic-local counterparts (:mod:`parsec_python.V_ion.slab`) use the same ``|z|`` kernel, so the total
electrostatics of a neutral slab is independent of the box height.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..Grid.pbc import SlabRealSpaceGrid
from ..Laplacian.finite_difference import second_derivative_coefficients
from ..Laplacian.lattice_metric import second_derivative_plan
from ..models import HartreeSettings
from .pbc import PeriodicHartreeResult
from .poisson import _conjugate_gradient


def _in_plane_wavevectors(grid: SlabRealSpaceGrid) -> np.ndarray:
    """``|G|`` of every in-plane Fourier index of the FFT of an ``n1 x n2`` plane."""
    n1, n2 = (int(v) for v in grid.points_per_axis[:2])
    reciprocal = 2.0 * np.pi * np.linalg.inv(grid.cell.lattice_vectors[:2, :2]).T  # rows b1, b2
    m1 = np.fft.fftfreq(n1, d=1.0 / n1)
    m2 = np.fft.fftfreq(n2, d=1.0 / n2)
    g = m1[:, None, None] * reciprocal[0][None, None, :] + m2[None, :, None] * reciprocal[1][None, None, :]
    return np.linalg.norm(g, axis=2)


def slab_boundary_potential(
    density: np.ndarray, grid: SlabRealSpaceGrid, layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Potential (Rydberg) at the ``layers`` planes beyond the bottom and the top plane of the box.

    Returns ``(below, above)``, each of shape ``(layers, n1, n2)``; ``below[s - 1]`` is the plane ``s`` steps
    under the first plane, ``above[s - 1]`` the plane ``s`` steps over the last one.
    """
    n1, n2, n3 = (int(v) for v in grid.points_per_axis)
    hz = grid.z_step
    rho = np.asarray(density, dtype=np.float64).reshape(n1, n2, n3)
    z_planes = grid.z_origin + hz * np.arange(n3)
    g = _in_plane_wavevectors(grid)  # (n1, n2)
    nonzero = g > 0.0
    inverse_g = np.zeros_like(g)
    inverse_g[nonzero] = 1.0 / g[nonzero]
    rho_g = np.fft.fft2(rho, axes=(0, 1)) / float(n1 * n2)  # (n1, n2, n3)
    sigma = hz * rho_g[0, 0, :].real  # charge per area of every plane (the G = 0 coefficient)

    def evaluate(z_values: np.ndarray) -> np.ndarray:
        out = np.empty((z_values.size, n1, n2))
        for index, z in enumerate(z_values):
            separation = np.abs(z - z_planes)  # (n3,)
            plane_average = -2.0 * np.pi * float(np.sum(sigma * separation))
            decay = np.exp(-g[:, :, None] * separation[None, None, :])  # (n1, n2, n3)
            coefficients = (2.0 * np.pi * hz) * inverse_g * np.sum(rho_g * decay, axis=2)
            coefficients[0, 0] = 0.0
            plane = np.fft.ifft2(coefficients * float(n1 * n2)).real
            out[index] = 2.0 * (plane + plane_average)  # Hartree -> Rydberg
        return out

    steps = np.arange(1, layers + 1)
    below = evaluate(z_planes[0] - steps * hz)
    above = evaluate(z_planes[-1] + steps * hz)
    return below, above


def solve_slab_hartree(
    density: np.ndarray,
    grid: SlabRealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
) -> PeriodicHartreeResult:
    """Hartree potential (Rydberg) of ``density`` on a slab grid: ``A V = 8 pi rho - (boundary terms)``."""
    if negative_laplacian.shape != (grid.size, grid.size):
        raise ValueError("negative_laplacian shape does not match the grid")
    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    n1, n2, n3 = (int(v) for v in grid.points_per_axis)
    width = grid.settings.stencil_half_width
    coefficients = second_derivative_coefficients(grid.settings.expansion_order)
    axial_z = float(second_derivative_plan(grid.step_vectors).axial[2])

    rhs = 8.0 * np.pi * density
    below, above = slab_boundary_potential(density, grid, width)
    rhs_box = rhs.reshape(n1, n2, n3)
    # A row on plane k misses the neighbor ``shell`` planes away when that plane is outside the box; the matrix
    # entry of a neighbor is -coefficient*axial_z, and its known value moves to the right-hand side.
    for shell in range(1, width + 1):
        entry = -coefficients[width + shell] * axial_z
        for plane in range(n3):
            top = plane + shell - (n3 - 1)  # 1.. : the layer over the last plane
            if top >= 1:
                rhs_box[:, :, plane] -= entry * above[top - 1]
            bottom = shell - plane  # 1.. : the layer under the first plane
            if bottom >= 1:
                rhs_box[:, :, plane] -= entry * below[bottom - 1]
    rhs = rhs_box.reshape(-1)

    if initial_potential is None:
        initial = np.zeros(grid.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_potential, dtype=np.float64)
        if initial.shape != (grid.size,):
            raise ValueError("initial Hartree potential does not match the grid")
    potential, converged, iterations, matvecs, residual, initial_residual = _conjugate_gradient(
        negative_laplacian, rhs, initial, settings
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            f"slab Hartree conjugate-gradient solve did not converge: residual={residual:.3e}, matvecs={matvecs}"
        )
    return PeriodicHartreeResult(
        potential=potential,
        right_hand_side=rhs,
        converged=converged,
        iterations=iterations,
        matrix_vector_products=matvecs,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
    )


__all__ = ["slab_boundary_potential", "solve_slab_hartree"]
