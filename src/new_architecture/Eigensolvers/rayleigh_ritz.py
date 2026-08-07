"""Rayleigh--Ritz projection and rotation for PARSEC eigensolvers.

Fortran provenance
------------------
PARSEC commit ``f432777750c2efc633adeb26faff641500b39b4a``.  The projection,
``my_dsyev`` diagonalization, and basis rotations translate the corresponding
sections of ``src/chebff.f90z`` and ``src/subspace.f90z``.  PARSEC supplies an
orthonormal basis before this stage; this primitive deliberately does not hide
an additional QR factorization.

For an orthonormal basis ``Q`` the method computes

``Y = H Q``

``H_sub = Q.H Y``

``H_sub U = U diag(lambda)``

``Psi = Q U``.

The full-space residual diagnostic for Ritz pair ``i`` is
``||H Psi_i - lambda_i Psi_i||_2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RayleighRitzResult:
    """Projected eigenpairs, cached ``H@Psi``, and full-space residuals."""

    eigenvalues: np.ndarray
    wavefunctions: np.ndarray
    applied_wavefunctions: np.ndarray
    projected_hamiltonian: np.ndarray
    residual_norms: np.ndarray


def rayleigh_ritz(operator: Any, basis: np.ndarray) -> RayleighRitzResult:
    """Perform PARSEC's Hermitian Rayleigh--Ritz projection and rotation.

    The caller must provide an orthonormal basis.  This primitive computes all
    residual norms and SUBSPACE exposes them as diagnostics.  CHEBFF discards
    them, and neither path adds an unrequested residual stop to PARSEC's
    fixed-cycle/fixed-pass algorithms.
    """

    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("basis must be a nonempty two-dimensional matrix")
    shape = getattr(operator, "shape", None)
    if shape is None or tuple(shape) != (basis.shape[0], basis.shape[0]):
        raise ValueError("operator shape must match the basis row dimension")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis must contain only finite values")

    # One block Hamiltonian application supplies Y=H@Q for both projection
    # and the later residual calculation.
    applied_basis = np.asarray(operator @ basis)
    if applied_basis.shape != basis.shape:
        raise ValueError("operator must preserve the basis shape")
    if not np.all(np.isfinite(applied_basis)):
        raise FloatingPointError("operator application produced nonfinite values")

    raw_projection = basis.conj().T @ applied_basis
    # PARSEC gives one stored triangle to a Hermitian eigensolver.  Mirror the
    # lower triangle explicitly so roundoff asymmetry in Q^H H Q cannot make
    # NumPy treat the small projected problem as non-Hermitian.
    lower = np.tril(raw_projection)
    projected_hamiltonian = lower + np.tril(lower, -1).conj().T
    eigenvalues, rotations = np.linalg.eigh(projected_hamiltonian)
    # Rotate Q and H@Q by the same small eigenvector matrix U.  Thus
    # applied_wavefunctions is H@(Q@U) without another costly H application.
    wavefunctions = basis @ rotations
    applied_wavefunctions = applied_basis @ rotations
    residuals = (
        applied_wavefunctions
        - wavefunctions * eigenvalues[np.newaxis, :]
    )
    residual_norms = np.linalg.norm(residuals, axis=0)

    return RayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=wavefunctions,
        applied_wavefunctions=applied_wavefunctions,
        projected_hamiltonian=projected_hamiltonian,
        residual_norms=residual_norms,
    )
