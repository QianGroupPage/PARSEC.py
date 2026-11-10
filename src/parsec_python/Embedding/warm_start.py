"""How the SCF loop of a moved FDET tip starts (PARSEC 1.5, ``parsec.F90:2815-2913``).

After a move of the tip atoms PARSEC continues from the last mixed potential
with the ionic potential exchanged (:class:`parsec_python.SCF.SCFWarmStart`).
PARSEC 1.5 adds two starts for FDET scans:

* **Translation** (``parsec.F90:2815-2899``, the classical mode): when every
  tip atom moved by the same whole number of grid steps ``S``, the density,
  the orbitals and the Hartree potential are translated by ``S`` on the grid
  (grid point ``k`` takes the value of point ``k - S``; points without a
  source get zero, the Hartree potential keeps its old value there).  The
  Hartree, exchange-correlation and non-additive kinetic potentials of the
  translated density then give the first input potential,
  ``V_in = V_ion + V_H[rho'] + V_xc[rho' + rho_core + rho_A] + v_nad``.
  1.5 does this in serial runs only; here it runs wherever the moved
  structure starts warm.
* **Tip sphere** (``parsec.F90:2903-2913``): the tip keeps its place on its
  own grid and the sample field moves under it, so the density and orbitals
  are kept and only the potentials are rebuilt from the last density with
  the new field.

Both use the guessed-density start of :func:`parsec_python.SCF.single_point.run_scf`
(``density_guess`` with ``hxc_reference`` zero and the bare new ionic
potential as ``input_potential``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

#: Positions that differ from a whole grid step by less than this count as one (``parsec.F90:2825``).
RIGID_TOLERANCE = 1.0e-6


def rigid_grid_shift(old_positions, new_positions, spacing: float) -> np.ndarray | None:
    """The whole grid steps ``S`` every atom moved by, ``None`` if they did not or ``S = 0``.

    ``parsec.F90:2817-2829``: ``S = nint((x_1 - x_1,old)/h)`` from the first
    atom, and every atom must have moved by ``S h`` to 1e-6 bohr.
    """

    old = np.asarray(old_positions, dtype=np.float64).reshape(-1, 3)
    new = np.asarray(new_positions, dtype=np.float64).reshape(-1, 3)
    if old.shape != new.shape or old.shape[0] == 0:
        return None
    from ..Grid.space_group import fortran_nint

    # Fortran's nint (halves away from zero); a half step fails the rigidity test below in any case.
    shift = np.array([fortran_nint((new[0, axis] - old[0, axis]) / float(spacing)) for axis in range(3)], dtype=np.int64)
    if not np.any(shift):
        return None
    if np.any(np.abs(new - old - shift * float(spacing)) > RIGID_TOLERANCE):
        return None
    return shift


def translation_rows(grid, shift) -> np.ndarray:
    """Row ``i`` of the translated field takes row ``rows[i]`` (``-1``: no source), ``k_i - S``."""

    return grid.rows_for_integer_coordinates(np.asarray(grid.integer_coordinates, dtype=np.int64) - np.asarray(shift))


def translate_field(values, rows: np.ndarray, fill=None) -> np.ndarray:
    """``values`` translated by ``rows``; zero (or ``fill`` at the same row) where there is no source."""

    values = np.asarray(values, dtype=np.float64)
    moved = np.zeros_like(values) if fill is None else np.array(fill, dtype=np.float64, copy=True)
    valid = rows >= 0
    moved[valid] = values[rows[valid]]
    return moved


def translate_subspace(state, rows: np.ndarray):
    """A saved host eigensubspace with its vectors translated on the grid, or ``None``.

    Handles :class:`parsec_python.Eigensolvers.EigvalState` (and its DIAGLA
    window); a state of another kind is not translated (``None``: the moved
    structure then starts its eigensolver afresh).
    """

    subspace = getattr(state, "subspace", None)
    vectors = getattr(subspace, "vectors", None)
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] != rows.size:
        return None
    valid = rows >= 0
    moved = np.zeros_like(vectors)
    moved[valid] = vectors[rows[valid]]
    updated = replace(state, subspace=replace(subspace, vectors=moved))
    window = getattr(state, "window_vectors", None)
    if isinstance(window, np.ndarray) and window.ndim == 2 and window.shape[0] == rows.size:
        shifted = np.zeros_like(window)
        shifted[valid] = window[rows[valid]]
        updated = replace(updated, window_vectors=shifted)
    return updated


def embedding_warm_start(previous_system, previous_result, system, warm_start, *, translate_state=None):
    """The FDET start of the SCF loop of ``system`` after a move, or ``warm_start`` unchanged.

    ``warm_start`` is the plain PARSEC warm start of the moved structure
    (:class:`parsec_python.SCF.SCFWarmStart`).  ``translate_state(rows)``,
    where given, translates the eigensubspace the solver keeps (an
    accelerated solver) and returns the state to hand on; without it a host
    state in ``warm_start`` is translated here.
    """

    field = getattr(system, "embedding", None)
    if field is None or warm_start is None or getattr(previous_result, "spin", None) is not None:
        return warm_start
    density = np.asarray(previous_result.density, dtype=np.float64)
    hartree = np.asarray(previous_result.hartree_potential, dtype=np.float64)
    zero = np.zeros_like(density)
    if getattr(field, "tip_sphere", None) is not None:
        # parsec.F90:2903-2913: density and orbitals kept, the sample field moved.
        return replace(
            warm_start,
            input_potential=np.array(system.ionic_potential, dtype=np.float64, copy=True),
            hartree_potential=hartree,
            density_guess=density,
            hxc_reference=zero,
            first_iteration_degree=None,
        )
    old = [atom.position for atom in previous_system.atoms]
    new = [atom.position for atom in system.atoms]
    shift = rigid_grid_shift(old, new, system.grid.spacing)
    if shift is None:
        return warm_start
    rows = translation_rows(system.grid, shift)
    if translate_state is not None:
        state = translate_state(rows)
    else:
        state = translate_subspace(warm_start.eigensolver_state, rows)
    return replace(
        warm_start,
        input_potential=np.array(system.ionic_potential, dtype=np.float64, copy=True),
        # The Hartree potential keeps its old value where the shifted one has no source.
        hartree_potential=translate_field(hartree, rows, fill=hartree),
        density_guess=translate_field(density, rows),
        hxc_reference=zero,
        eigensolver_state=state,
        first_iteration_degree=None,
    )


__all__ = [
    "RIGID_TOLERANCE",
    "embedding_warm_start",
    "rigid_grid_shift",
    "translate_field",
    "translate_subspace",
    "translation_rows",
]
