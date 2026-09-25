"""Complex Bloch (k-point) Kohn-Sham Hamiltonian for periodic calculations.

For a periodic effective potential, Bloch's theorem writes each orbital as
``psi_nk(r) = e^{ik.r} u_nk(r)`` with ``u_nk`` periodic.  Substituting into
the real-space ``H = -nabla^2 + V_eff + V_NL`` (Rydberg units) and factoring
out the phase gives the transformed operator acting on ``u_nk`` alone,

``H_k = e^{-ik.r} H e^{ik.r}
      = -nabla^2 - 2i*k.grad + |k|^2 + V_eff + V_NL,k``,

where ``V_NL,k`` is the ordinary KB nonlocal operator built with Bloch-phase-
weighted periodic images
(:func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`'s
``k_point`` argument).  At ``k=(0,0,0)`` this reduces exactly to the
existing real Gamma-point :class:`~parsec_python.Hamiltonian.operator.KohnShamHamiltonian`
(verified directly: identical projector values, and the ``-2i*k.grad+|k|^2``
terms vanish).

This module intentionally does not include spin-orbit coupling; combining
periodic k-point sampling with SOC needs the k-point-phase-weighted spin-
orbit projectors (:mod:`~parsec_python.Eigensolvers.perturbative_soc`
currently only builds isolated, single-copy ones) and is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..V_ion import NonlocalProjectorOperator


def _as_block(vectors: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(vectors, dtype=complex)
    if array.ndim == 1:
        return array[:, None], True
    return array, False


@dataclass(frozen=True)
class KPointKohnShamHamiltonian:
    """``H_k`` acting on the complex periodic Bloch factor ``u_nk``.

    ``negative_laplacian`` and ``gradient`` are grid/spacing-derived
    operators shared by every k-point
    (:func:`~parsec_python.Laplacian.build_negative_laplacian`,
    :func:`~parsec_python.Laplacian.build_gradient`); ``nonlocal_operator``
    must have been built with this same ``k_point`` (its projectors carry
    the Bloch phase).  ``effective_potential`` is the ordinary real,
    k-independent local field ``V_ion,local + V_H + V_xc``.
    """

    negative_laplacian: sp.csr_matrix
    gradient: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    effective_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    k_point: np.ndarray

    def __post_init__(self) -> None:
        potential = np.asarray(self.effective_potential, dtype=float)
        size = self.negative_laplacian.shape[0]
        if self.negative_laplacian.shape != (size, size):
            raise ValueError("negative_laplacian must be square")
        if any(g.shape != (size, size) for g in self.gradient):
            raise ValueError("gradient operators must match negative_laplacian's shape")
        if potential.shape != (size,):
            raise ValueError("effective potential does not match the kinetic operator")
        if self.nonlocal_operator.shape != (size, size):
            raise ValueError("nonlocal operator does not match the kinetic operator")
        k_point = np.asarray(self.k_point, dtype=float)
        if k_point.shape != (3,):
            raise ValueError("k_point must have shape (3,)")
        object.__setattr__(self, "effective_potential", potential)
        object.__setattr__(self, "gradient", tuple(self.gradient))
        object.__setattr__(self, "k_point", k_point)

    @property
    def shape(self) -> tuple[int, int]:
        return self.negative_laplacian.shape

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply ``H_k = -nabla^2 - 2i*k.grad + |k|^2 + V_eff + V_NL,k``."""
        block, was_1d = _as_block(vectors)
        result = self.negative_laplacian @ block
        k_squared = float(np.dot(self.k_point, self.k_point))
        for axis, gradient_axis in enumerate(self.gradient):
            k_component = self.k_point[axis]
            if k_component == 0.0:
                continue
            result = result - (2j * k_component) * (gradient_axis @ block)
        if k_squared != 0.0:
            result = result + k_squared * block
        result = result + self.effective_potential[:, None] * block
        result = result + self.nonlocal_operator.apply(block)
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


__all__ = ["KPointKohnShamHamiltonian"]
