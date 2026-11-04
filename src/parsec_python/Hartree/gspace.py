"""``Calc_Hartree_G``: the periodic Hartree potential solved in reciprocal space.

PARSEC's keyword ``Calc_Hartree_G`` (``usrinputfile.F90`` l.907-912, 3-D
periodic cells only) sends ``rho_har`` (``rho_hart.F90`` l.108-113) to
``hpot_G`` (``hpot_g.F90``) instead of the conjugate-gradient solve of the
finite-difference Poisson equation.  The routine intends

``V_H(G) = 8 pi rho(G) / |G|^2``,  ``V_H(G = 0) = 0``  (Rydberg),

on the FFT box of the cell, but it is broken: it prints "IT WILL NOT WORK",
it averages ``rho(G)`` over the symmetry stars into ``rho_g(star)`` and then
reads that array with the index of the single G-vector (``rho_g(ig)``,
``ig = 2..ng``, l.198-205), so every G beyond the first stars takes the
density of another (or no) star, and the Hartree energy it prints is never
used.  Fortran's numbers with the keyword are therefore not a reference.

This module solves the intended equation exactly on the periodic grid of the
run: the density is transformed by a 3-D FFT over the ``n1 x n2 x n3``
points, every Fourier coefficient with the integer index ``m`` (``numpy``
``fftfreq`` order, ``-n/2 <= m < n/2``) is divided by
``|G_m|^2 = |m_1 b_1 + m_2 b_2 + m_3 b_3|^2`` with the reciprocal lattice
vectors ``b_i = 2 pi inv(A)^T`` of the cell (rows), ``G = 0`` is set to
zero, and the inverse FFT gives ``V_H``.  The grid shift only multiplies
each coefficient by a phase that the inverse transform removes, so the same
code serves the orthorhombic grid and Fortran's general (non-orthogonal)
grid.  With an even number of points along a direction the Nyquist index
``-n/2`` stands for both ``+-n/2``; on a non-orthogonal cell their ``|G|``
differ, and the real part of the inverse transform takes the mean of the
two ``1/|G|^2`` (exact on an orthorhombic cell, where they agree).

Unlike the finite-difference solve (``hpotcg``, the default, unchanged), this
``V_H`` is the spectral solution: it has no finite-difference error at large
``|G|``.  Units: density in e/bohr^3, potential in Ry.
"""

from __future__ import annotations

import numpy as np

from ..Grid.cell import reciprocal_lattice


def g_squared(grid) -> np.ndarray:
    """``|G_m|^2`` (bohr^-2) of every FFT index of ``grid``, shape ``grid.shape``."""

    shape = tuple(int(value) for value in grid.points_per_axis)
    reciprocal = reciprocal_lattice(np.asarray(grid.cell.lattice_vectors, dtype=float))
    indices = np.meshgrid(
        *(np.fft.fftfreq(n, d=1.0 / n) for n in shape), indexing="ij"
    )
    vectors = sum(
        index[..., None] * reciprocal[axis][None, None, None, :]
        for axis, index in enumerate(indices)
    )
    return np.sum(vectors**2, axis=-1)


def solve_hartree_g_space(density: np.ndarray, grid) -> np.ndarray:
    """``V_H`` (Ry) of ``density`` (e/bohr^3) on a periodic grid, by FFT.

    ``V_H(G) = 8 pi rho(G)/|G|^2`` for ``G != 0`` and zero at ``G = 0``, so
    the potential has zero mean and the neutralizing background of the cell
    is implied (``rho - <rho>`` and ``rho`` give the same potential).
    """

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    shape = tuple(int(value) for value in grid.points_per_axis)
    transform = np.fft.fftn(density.reshape(shape))
    squared = g_squared(grid)
    squared[0, 0, 0] = 1.0
    transform *= 8.0 * np.pi / squared
    transform[0, 0, 0] = 0.0
    potential = np.real(np.fft.ifftn(transform)).reshape(-1)
    return potential - float(np.mean(potential))


__all__ = ["g_squared", "solve_hartree_g_space"]
