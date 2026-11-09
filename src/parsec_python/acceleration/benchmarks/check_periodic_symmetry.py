"""Time the symmetry machinery of a bulk-diamond supercell: space group, grid group, density table and average.

``python -m parsec_python.acceleration.benchmarks.check_periodic_symmetry [n ...]`` builds the cube of ``n^3``
conventional cells (8 n^3 atoms, coordinates written with 12 decimals as an input file holds them) on the grid of
the diamond benchmark (18 points per edge, half a step off the origin) and prints the times of
:func:`~parsec_python.Grid.space_group.find_space_group`, of
:func:`~parsec_python.Grid.space_group.find_grid_space_group`, of the density table
(:func:`~parsec_python.Grid.space_group.periodic_grid_symmetry`) and of one average on the host
(:func:`~parsec_python.Grid.point_group.symmetrize_density`) and, where CuPy runs, on the device
(:class:`~parsec_python.acceleration.SCF.periodic_symmetry.PeriodicDensitySymmetrizer`, with the affine maps and
with the uploaded table), each compared with the host bit for bit.  Default ``n``: 4 and 8 (4,096 atoms, 3 M points).
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Grid.point_group import symmetrize_density
from parsec_python.Grid.space_group import find_grid_space_group, find_space_group, periodic_grid_symmetry
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_A = 3.567 / 0.529177210903
_BASIS = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


def diamond(n: int):
    """The lattice and the 8 n^3 atoms of n^3 conventional cells (bohr, 12 decimals)."""

    atoms = [
        Atom("C", [round((c + b + s) * _A, 12) for c, b in zip((i, j, k), basis)])
        for i in range(n) for j in range(n) for k in range(n) for s in (0.0, 0.25) for basis in _BASIS
    ]
    return np.diag([round(n * _A, 10)] * 3), atoms


def main(argv=None) -> int:
    sizes = [int(value) for value in (argv if argv is not None else sys.argv[1:])] or [4, 8]
    try:
        import cupy
    except ImportError:  # pragma: no cover - depends on the machine
        cupy = None
    from parsec_python.acceleration.SCF.periodic_symmetry import PeriodicDensitySymmetrizer

    for n in sizes:
        lattice, atoms = diamond(n)
        settings = PeriodicGridSettings(
            spacing=_A / 18 * (1.0 - 1.0e-9), expansion_order=8, shift=(0.5, 0.5, 0.5), points_rule="fortran"
        )
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)
        start = time.perf_counter()
        crystal = find_space_group(lattice, atoms)
        searched = time.perf_counter()
        group = find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
        grouped = time.perf_counter()
        symmetry = periodic_grid_symmetry(grid, group)
        tabled = time.perf_counter()
        density = np.random.default_rng(1).random(grid.size)
        host, _ = symmetrize_density(density, symmetry, grid.volume_element)
        averaged = time.perf_counter()
        print(
            f"n={n}: {len(atoms)} atoms, {grid.size} points; space group {crystal.order} in {searched - start:.2f} s, "
            f"grid group {group.order} in {grouped - searched:.2f} s, table {tabled - grouped:.2f} s "
            f"({symmetry.images.nbytes / 1e9:.2f} GB), host average {averaged - tabled:.3f} s",
            flush=True,
        )
        if cupy is None:
            continue
        for label, source in (("affine maps", symmetry), ("uploaded table", replace(symmetry, affine_maps=None, counts=None))):
            averager = PeriodicDensitySymmetrizer(source, xp=cupy)
            field = cupy.asarray(density)
            result, _ = averager.symmetrize_density(field, grid.volume_element)
            cupy.cuda.Device().synchronize()
            begin = time.perf_counter()
            for _ in range(5):
                result, _ = averager.symmetrize_density(field, grid.volume_element)
            cupy.cuda.Device().synchronize()
            seconds = (time.perf_counter() - begin) / 5
            same = bool(np.array_equal(cupy.asnumpy(result), host))
            pool = cupy.get_default_memory_pool()
            print(f"   device ({label}): {seconds * 1e3:.1f} ms per average, the host's bits: {same}, "
                  f"pool {pool.total_bytes() / 1e9:.2f} GB", flush=True)
            del averager, field, result
            pool.free_all_blocks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
