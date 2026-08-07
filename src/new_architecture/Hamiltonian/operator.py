"""Composition and application of the real-space Kohn--Sham Hamiltonian.

For the supported real, scalar, isolated calculation, define

``A = -nabla_FD^2``

``V_eff = V_ion,local + V_H + V_xc``

``V_NL = B*diag(sign(D))*B.T``.

The discrete Hamiltonian action on one orbital or a block ``Q`` is

``H Q = A Q + V_eff[:,None]*Q + B*diag(sign(D))*(B.T@Q)``.

``V_eff`` contains only grid-diagonal potentials; the KB nonlocal ionic term
is deliberately kept separate.  Production eigensolvers use this composed
action through a ``LinearOperator`` instead of materializing a full
grid-by-grid Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ..V_ion import NonlocalProjectorOperator


@dataclass(frozen=True)
class KohnShamHamiltonian:
    """Bind the current local effective potential to the static operators.

    ``negative_laplacian`` and ``nonlocal_operator`` depend only on the grid,
    nuclei, and pseudopotentials, so a single-point calculation builds them
    once.  ``effective_potential`` is the current SCF input field

    ``V_eff^in = V_ion,local + V_H^in + V_xc^in``.

    A new lightweight instance is made when that local field changes.  In
    Rydberg units the kinetic operator is ``-nabla^2`` rather than the
    Hartree-unit ``-nabla^2/2``.
    """

    negative_laplacian: sp.csr_matrix
    effective_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator

    def __post_init__(self) -> None:
        # This class intentionally represents PARSEC's real scalar path.  The
        # float conversion is not suitable for complex k-point or spin-orbit
        # wavefunctions, which are outside the present implementation scope.
        potential = np.asarray(self.effective_potential, dtype=float)
        size = self.negative_laplacian.shape[0]
        if self.negative_laplacian.shape != (size, size):
            raise ValueError("negative_laplacian must be square")
        if potential.shape != (size,):
            raise ValueError("effective potential does not match the kinetic operator")
        if self.nonlocal_operator.shape != (size, size):
            raise ValueError("nonlocal operator does not match the kinetic operator")
        object.__setattr__(self, "effective_potential", potential)

    @property
    def shape(self) -> tuple[int, int]:
        return self.negative_laplacian.shape

    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the complete central and off-diagonal FD ``-nabla^2``."""
        return np.asarray(self.negative_laplacian @ vectors)

    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        """Apply ``diag(V_eff)`` by pointwise multiplication.

        For a block ``vectors.shape == (n_grid, n_vectors)``, ``[:, None]``
        broadcasts the same grid potential over every orbital column.
        """
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            return self.effective_potential * vectors
        return self.effective_potential[:, None] * vectors

    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the low-rank KB term ``B*diag(sign(D))*B.T``."""
        return self.nonlocal_operator.apply(vectors)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply all Hamiltonian terms without constructing their full sum."""
        return (
            self.apply_kinetic(vectors)
            + self.apply_local(vectors)
            + self.apply_nonlocal(vectors)
        )

    def as_linear_operator(self) -> LinearOperator:
        """Expose the production matrix-vector/block action to eigensolvers.

        Lanczos spectral estimates and every Chebyshev-filter application
        enter through these callbacks, so they all use exactly the same
        kinetic, local, and nonlocal composition.
        """
        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=float,
        )

    def as_sparse(self) -> sp.csr_matrix:
        """Materialize ``A + diag(V_eff) + V_NL`` for small diagnostics.

        Forming ``V_NL = B*diag(sign(D))*B.T`` can be much denser than storing
        its projector factors.  The SCF/eigensolver path therefore does not
        call this method.
        """
        return (
            self.negative_laplacian
            + sp.diags(self.effective_potential, format="csr")
            + self.nonlocal_operator.as_sparse()
        ).tocsr()
