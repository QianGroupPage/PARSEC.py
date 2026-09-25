"""Complex two-component spinor Kohn-Sham Hamiltonian (self-consistent SOC).

This is the matrix-free grid operator behind Fortran PARSEC's
``SO_from_scratch``/``elec_st%is_so`` path: unlike
:mod:`~parsec_python.Eigensolvers.perturbative_soc` (which only ever
contracts the spin-orbit projectors against a *fixed, small* basis of
already-converged states to build a ``2N x 2N`` matrix), this module applies
the same spin-orbit projector pair as a genuine operator on arbitrary
grid-space spinors, so it can sit inside an iterative eigensolver and be
part of the self-consistent field from the first iteration.

A spinor ``Psi = (psi_up; psi_down)`` is represented as a complex array of
shape ``(2*n_grid,)`` or ``(2*n_grid, n_vectors)``: rows ``0:n_grid`` are the
spin-up component, rows ``n_grid:2*n_grid`` are spin-down, matching
``zmatvec_so.f90``'s ``p(1:dim)``/``p(1+dim:dim2)`` split.  The action is

``H*Psi = [T + diag(V_eff) + V_NL]*Psi (per channel, independently)``
        ``+ lzsz(psi_up)  -> q_up   (spin-diagonal)``
        ``+ lzsz(psi_down) -> q_down (spin-diagonal)``
        ``+ lsxy(psi_down, m=-1) -> q_up   (spin-off-diagonal)``
        ``+ lsxy(psi_up,   m=+1) -> q_down (spin-off-diagonal)``

exactly mirroring ``zmatvec_so.f90``'s call pattern.  ``lzsz``/``lsxy``'s
grid-space application (``apply_lzsz``/``apply_lsxy`` below) is built from
the *same* per-atom projector overlaps
(:func:`~parsec_python.Eigensolvers.perturbative_soc._projector_overlaps`)
already validated in the perturbative path -- the two modules share that one
overlap primitive so their formulas cannot silently drift apart.

This module deliberately does not yet support non-collinear magnetism's
``B_xc.sigma`` term (Fortran's ``Non_Collinear_magnetism`` flag): both spinor
channels share one spin-unpolarized effective potential ``V_eff``, matching
``SO_from_scratch`` without ``Non_Collinear_magnetism`` -- the simplest and
most common self-consistent-SOC configuration (SOC band splitting on a
non-magnetic system), not a combination with collinear/non-collinear
magnetism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..Eigensolvers.perturbative_soc import (
    _LM_TO_L,
    _LM_TO_ML,
    AtomSpinOrbitProjectors,
    _projector_overlaps,
)
from ..V_ion import NonlocalProjectorOperator


def _as_block(vectors: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``vectors`` as a 2D ``(n, k)`` complex array plus whether the
    caller's input was 1D (so the result can be squeezed back)."""
    array = np.asarray(vectors, dtype=complex)
    if array.ndim == 1:
        return array[:, None], True
    return array, False


def _apply_nonlocal_complex(
    nonlocal_operator: NonlocalProjectorOperator, vectors: np.ndarray
) -> np.ndarray:
    """dtype-preserving reimplementation of ``NonlocalProjectorOperator.apply``.

    The production method forces ``dtype=float`` internally (correct for the
    real scalar/spin-polarized paths, but silently discards the imaginary
    part of a complex spinor).  The ordinary KB projectors themselves are
    real-valued (real spherical harmonics), so applying them to a complex
    vector is just an ordinary real-sparse-times-complex-dense product; only
    the dtype needs to be preserved, not any new physics.
    """
    coefficients = nonlocal_operator.projectors.T @ vectors
    coefficients = nonlocal_operator.signs[:, None] * coefficients
    return np.asarray(nonlocal_operator.projectors @ coefficients)


def apply_lzsz(
    projectors: Sequence[AtomSpinOrbitProjectors],
    wavefunctions: np.ndarray,
    spin_sign: int,
) -> np.ndarray:
    """Grid-space ``q = lzsz_sigma(psi)`` for one spin channel (``zmatvec_so.f90``).

    ``wavefunctions`` is ``(n_grid, n_vectors)`` complex, all in the *same*
    spin channel; ``spin_sign`` is +1 (up) or -1 (down).  Returns the
    spin-diagonal spin-orbit contribution to that same channel.
    """
    block, was_1d = _as_block(wavefunctions)
    n_grid = block.shape[0]
    output = np.zeros_like(block)
    if not projectors:
        return output[:, 0] if was_1d else output
    dotio_by_atom, dotso_by_atom = _projector_overlaps(projectors, block)
    lz = spin_sign * 0.5 * _LM_TO_ML
    quarter_ll1 = 0.25 * _LM_TO_L * (_LM_TO_L + 1)
    for atom, dotio, dotso in zip(projectors, dotio_by_atom, dotso_by_atom):
        # dotio, dotso: (n_vectors, 8); coeff_*: (8, n_vectors) to match
        # atom.v_ion/v_so's (n_support, 8) for a (n_support,8)@(8,n_vectors)
        # contraction.
        coeff_ion = (lz[None, :] * dotso).T
        coeff_so = (
            lz[None, :] * dotio - 0.5 * lz[None, :] * dotso + quarter_ll1[None, :] * dotso
        ).T
        contribution = atom.v_ion @ coeff_ion + atom.v_so @ coeff_so
        output[atom.support_rows, :] += contribution
    return output[:, 0] if was_1d else output


def apply_lsxy(
    projectors: Sequence[AtomSpinOrbitProjectors],
    wavefunctions_in: np.ndarray,
    spin_sign: int,
) -> np.ndarray:
    """Grid-space ``q = lsxy_m(psi_in)`` (``zmatvec_so.f90``/``pls.F90``).

    ``wavefunctions_in`` lives in one spin channel; the *caller* adds the
    result into the opposite channel, matching ``zmatvec_so.f90``'s
    ``lsxy(...,pup,qdn,m=1,...)``/``lsxy(...,pdn,qup,m=-1,...)`` pattern.
    ``spin_sign`` is ``pls.F90``'s ladder-direction ``m`` argument: ``m=-1``
    (down input) produces the up-channel contribution, ``m=+1`` (up input)
    produces the down-channel contribution.
    """
    block, was_1d = _as_block(wavefunctions_in)
    output = np.zeros_like(block)
    if not projectors:
        return output[:, 0] if was_1d else output
    dotio_by_atom, dotso_by_atom = _projector_overlaps(projectors, block)
    for atom, dotio, dotso in zip(projectors, dotio_by_atom, dotso_by_atom):
        for lm in range(8):
            target = lm + spin_sign
            if target < 0 or target > 7 or _LM_TO_L[target] != _LM_TO_L[lm]:
                continue
            l = int(_LM_TO_L[lm])
            m_l = int(_LM_TO_ML[lm])
            number = l * (l + 1) - m_l * (spin_sign + m_l)
            if number <= 0:
                continue
            weight = 0.5 * np.sqrt(float(number))
            dotso_j = dotso[:, lm]
            dotio_j = dotio[:, lm]
            contribution = np.outer(
                atom.v_ion[:, target], weight * dotso_j
            ) + np.outer(atom.v_so[:, target], weight * (dotio_j - 0.5 * dotso_j))
            output[atom.support_rows, :] += contribution
    return output[:, 0] if was_1d else output


@dataclass(frozen=True)
class SpinorKohnShamHamiltonian:
    """The self-consistent-SOC Hamiltonian acting on stacked complex spinors.

    ``negative_laplacian`` and ``nonlocal_operator`` are the same
    grid/pseudopotential-derived operators the real scalar path already
    builds; ``effective_potential`` is the *single*, spin-unpolarized local
    field ``V_ion,local + V_H + V_xc`` shared by both spinor channels (see
    the module docstring on non-collinear magnetism's exclusion).
    ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output; an empty list degenerates ``H`` to the ordinary scalar
    Hamiltonian applied identically to both (otherwise uncoupled) channels.
    """

    negative_laplacian: sp.csr_matrix
    effective_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...]

    def __post_init__(self) -> None:
        potential = np.asarray(self.effective_potential, dtype=float)
        size = self.negative_laplacian.shape[0]
        if self.negative_laplacian.shape != (size, size):
            raise ValueError("negative_laplacian must be square")
        if potential.shape != (size,):
            raise ValueError("effective potential does not match the kinetic operator")
        if self.nonlocal_operator.shape != (size, size):
            raise ValueError("nonlocal operator does not match the kinetic operator")
        object.__setattr__(self, "effective_potential", potential)
        object.__setattr__(self, "soc_projectors", tuple(self.soc_projectors))

    @property
    def grid_size(self) -> int:
        return self.negative_laplacian.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape of the stacked ``(psi_up; psi_down)`` operator."""
        return (2 * self.grid_size, 2 * self.grid_size)

    def apply_channel(self, vectors: np.ndarray) -> np.ndarray:
        """Ordinary scalar-relativistic ``[T + diag(V_eff) + V_NL]`` on one channel."""
        block, was_1d = _as_block(vectors)
        result = (
            self.negative_laplacian @ block
            + self.effective_potential[:, None] * block
            + _apply_nonlocal_complex(self.nonlocal_operator, block)
        )
        return result[:, 0] if was_1d else result

    def apply(self, spinor: np.ndarray) -> np.ndarray:
        """Apply the full spinor Hamiltonian to stacked ``(2*n_grid, ...)`` input."""
        block, was_1d = _as_block(spinor)
        n_grid = self.grid_size
        if block.shape[0] != 2 * n_grid:
            raise ValueError(
                f"spinor input must have 2*n_grid={2 * n_grid} rows, got {block.shape[0]}"
            )
        psi_up = block[:n_grid, :]
        psi_down = block[n_grid:, :]

        q_up = self.apply_channel(psi_up)
        q_down = self.apply_channel(psi_down)

        if self.soc_projectors:
            q_up = q_up + apply_lzsz(self.soc_projectors, psi_up, spin_sign=1)
            q_up = q_up + apply_lsxy(self.soc_projectors, psi_down, spin_sign=-1)
            q_down = q_down + apply_lzsz(self.soc_projectors, psi_down, spin_sign=-1)
            q_down = q_down + apply_lsxy(self.soc_projectors, psi_up, spin_sign=1)

        result = np.concatenate([q_up, q_down], axis=0)
        return result[:, 0] if was_1d else result

    def as_linear_operator(self) -> "LinearOperator":
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=complex,
        )


__all__ = [
    "SpinorKohnShamHamiltonian",
    "apply_lzsz",
    "apply_lsxy",
]
