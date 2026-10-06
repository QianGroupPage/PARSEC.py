"""Complex Bloch spinor Kohn-Sham Hamiltonian: periodic self-consistent SOC.

Combines :class:`~parsec_python.Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian`
(the k-point kinetic/local/ordinary-nonlocal terms, reused unchanged and
applied independently to each spinor component) with
:func:`~parsec_python.Hamiltonian.spinor_operator.apply_lzsz`/``apply_lsxy``
(the spin-orbit cross terms, also reused unchanged -- they only depend on
the supplied projector pair, not on whether that pair was built for an
isolated atom or with Bloch-phase-weighted periodic images).  This mirrors
:class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`
exactly, with the plain ``KohnShamHamiltonian`` per-channel operator swapped
for ``KPointKohnShamHamiltonian`` and the SOC projectors built with this
same k-point's Bloch phase
(:func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
``k_point`` argument).

As with :mod:`~parsec_python.Hamiltonian.spinor_operator`, both spinor
components share one spin-unpolarized ``V_eff`` unless ``xc_delta`` is given:
that adds the diagonal collinear Zeeman-like term ``+diag(xc_delta)`` to the
up component and ``-diag(xc_delta)`` to the down component
(``xc_delta = (V_xc,up - V_xc,down)/2``), exactly as in the cluster
:class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`.
It is real and k-independent, so it needs no Bloch phase.  Non-collinear
magnetism (an off-diagonal ``B_xc.sigma`` term) is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from .kpoint_operator import KPointKohnShamHamiltonian
from .spinor_operator import _as_block, apply_lsxy, apply_lzsz


@dataclass(frozen=True)
class KPointSpinorKohnShamHamiltonian:
    """``H_k`` acting on a stacked complex spinor Bloch factor ``(u_up,k; u_down,k)``.

    ``scalar_hamiltonian`` must have been built with this same k-point (its
    nonlocal operator carries that k-point's Bloch phase, see
    :func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`'s
    ``k_point`` argument); ``soc_projectors`` must likewise come from
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`
    with that same ``k_point``.
    """

    scalar_hamiltonian: KPointKohnShamHamiltonian
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...]
    xc_delta: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "soc_projectors", tuple(self.soc_projectors))
        if self.xc_delta is not None:
            xc_delta = np.asarray(self.xc_delta, dtype=float)
            if xc_delta.shape != (self.scalar_hamiltonian.shape[0],):
                raise ValueError("xc_delta does not match the kinetic operator")
            object.__setattr__(self, "xc_delta", xc_delta)

    @property
    def grid_size(self) -> int:
        return self.scalar_hamiltonian.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (2 * self.grid_size, 2 * self.grid_size)

    def apply(self, spinor: np.ndarray) -> np.ndarray:
        block, was_1d = _as_block(spinor)
        n_grid = self.grid_size
        if block.shape[0] != 2 * n_grid:
            raise ValueError(
                f"spinor input must have 2*n_grid={2 * n_grid} rows, got {block.shape[0]}"
            )
        psi_up = block[:n_grid, :]
        psi_down = block[n_grid:, :]

        q_up = self.scalar_hamiltonian.apply(psi_up)
        q_down = self.scalar_hamiltonian.apply(psi_down)

        if self.soc_projectors:
            q_up = q_up + apply_lzsz(self.soc_projectors, psi_up, spin_sign=1)
            q_up = q_up + apply_lsxy(self.soc_projectors, psi_down, spin_sign=-1)
            q_down = q_down + apply_lzsz(self.soc_projectors, psi_down, spin_sign=-1)
            q_down = q_down + apply_lsxy(self.soc_projectors, psi_up, spin_sign=1)

        if self.xc_delta is not None:
            q_up = q_up + self.xc_delta[:, None] * psi_up
            q_down = q_down - self.xc_delta[:, None] * psi_down

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


__all__ = ["KPointSpinorKohnShamHamiltonian"]
