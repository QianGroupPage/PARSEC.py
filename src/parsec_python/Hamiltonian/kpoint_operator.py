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

This module holds scalar orbitals only.  Spin-orbit coupling with k-points
uses two-component spinors: :mod:`~parsec_python.Hamiltonian.kpoint_spinor_operator`
(the operator, with the Bloch-phase-weighted spin-orbit projectors of
:mod:`~parsec_python.Eigensolvers.perturbative_soc`) and
:mod:`~parsec_python.SCF.kpoints_soc` (the SCF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    ``kinetic_potential`` is the ``vtau`` of a meta-GGA
    (:mod:`parsec_python.V_xc.meta_gga`), ``None`` otherwise: the term
    ``sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c)`` with ``G`` = ``gradient``,
    ``e^{-ik.r} (-div(vtau grad)) e^{ik.r}`` on the grid (Hermitian, the
    derivative of the discrete meta-GGA energy with respect to ``u``).
    ``gradient_flux`` is ``W = de/dg`` (three fields) of the orbital form of
    the density gradient, ``g = (d/h^3) sum f 2 Re[conj(u) (G + ik) u]``
    (:func:`parsec_python.V_xc.meta_gga.meta_gga_gradient`): the term
    ``2 sum_c [W_c (G_c + i k_c) + (G_c + i k_c)^H W_c] = 2 sum_c [W_c G_c + G_c^T W_c]``
    (the ``k`` parts cancel for a real ``W``); ``None`` in the density form.
    ``sum_c k_c G_c`` (``_k_gradient``, one real sparse matrix, ``None`` at
    Gamma) is built on first use; ``k_gradient_cache`` is a dict that keeps it
    for the next Hamiltonians of the same ``k_point`` and ``gradient`` (the SCF
    drivers hold one per k-point, so that no iteration sums the three sparse
    matrices again, and a device kernel that never reads the matrix never
    builds it).
    """

    negative_laplacian: sp.csr_matrix
    gradient: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    effective_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    k_point: np.ndarray
    kinetic_potential: np.ndarray | None = None
    gradient_flux: tuple | None = None
    k_gradient_cache: dict | None = field(default=None, repr=False, compare=False)
    # V_eff + |k|^2, built once.
    _diagonal: np.ndarray = field(init=False, repr=False, compare=False)

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
        object.__setattr__(
            self, "_diagonal", potential + float(np.dot(k_point, k_point))
        )
        if self.kinetic_potential is not None:
            kinetic = np.ascontiguousarray(self.kinetic_potential, dtype=np.float64)
            if kinetic.shape != (size,):
                raise ValueError("meta-GGA vtau does not match the kinetic operator")
            object.__setattr__(self, "kinetic_potential", kinetic)
        if self.gradient_flux is not None:
            if self.kinetic_potential is None:
                raise ValueError("the meta-GGA gradient flux needs its vtau")
            fluxes = tuple(np.ascontiguousarray(flux, dtype=np.float64) for flux in self.gradient_flux)
            if len(fluxes) != 3 or any(flux.shape != (size,) for flux in fluxes):
                raise ValueError("the meta-GGA gradient flux must be three fields of the grid")
            object.__setattr__(self, "gradient_flux", fluxes)

    @property
    def _k_gradient(self) -> sp.csr_matrix | None:
        """``sum_c k_c G_c`` as one real sparse matrix (``None`` at Gamma), built on first use."""
        cache = self.k_gradient_cache
        if cache is None:
            cache = self.__dict__.setdefault("_own_k_gradient", {})
        if "matrix" not in cache:
            k_gradient = None
            for axis, gradient_axis in enumerate(self.gradient):
                if self.k_point[axis] != 0.0:
                    term = self.k_point[axis] * gradient_axis
                    k_gradient = term if k_gradient is None else k_gradient + term
            cache["matrix"] = None if k_gradient is None else k_gradient.tocsr()
        return cache["matrix"]

    def apply_kinetic_potential(self, block: np.ndarray) -> np.ndarray:
        """The meta-GGA terms of a complex block: ``sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c) u``
        and, in the orbital form, ``2 sum_c [W_c G_c u + G_c^T (W_c u)]``."""
        vtau = self.kinetic_potential[:, None]
        result = None
        for axis, gradient_axis in enumerate(self.gradient):
            k_axis = float(self.k_point[axis])
            derivative = (gradient_axis @ block.view(np.float64)).view(np.complex128)
            flux = vtau * derivative
            if k_axis != 0.0:
                flux += (1j * k_axis) * vtau * block
            extra = None
            if self.gradient_flux is not None:
                weight = self.gradient_flux[axis][:, None]
                extra = 2.0 * weight * derivative
                outgoing = flux + 2.0 * weight * block
            else:
                outgoing = flux
            outgoing = np.ascontiguousarray(outgoing)
            term = (gradient_axis.T @ outgoing.view(np.float64)).view(np.complex128)
            if k_axis != 0.0:
                term -= (1j * k_axis) * flux
            if extra is not None:
                term += extra
            result = term if result is None else result + term
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self.negative_laplacian.shape

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply ``H_k = -nabla^2 - 2i*k.grad + |k|^2 + V_eff + V_NL,k`` (+ the meta-GGA term)."""
        block, was_1d = _as_block(vectors)
        block = np.ascontiguousarray(block)
        # The Laplacian and gradient are real: applying them to the real view
        # (n, 2*n_vectors) of the complex block halves the sparse-product work
        # and avoids upcasting the matrices.
        real_view = block.view(np.float64)
        result = (self.negative_laplacian @ real_view).view(np.complex128)
        if self._k_gradient is not None:
            result -= 2j * (self._k_gradient @ real_view).view(np.complex128)
        result += self._diagonal[:, None] * block
        result += self.nonlocal_operator.apply(block)
        if self.kinetic_potential is not None:
            result += self.apply_kinetic_potential(block)
        return result[:, 0] if was_1d else result

    @property
    def spectral_margin(self) -> float:
        """Gershgorin's bound of the meta-GGA terms (Ry; 0 without), for the filter's upper bound.

        The row sums of :attr:`parsec_python.V_xc.meta_gga.KineticPotentialOperator.spectral_margin`
        with this k-point (``|G_c + ik_c| <= |G_c| + |k_c|`` entrywise):
        a narrow spike of ``vtau`` (vacuum of a slab or wire) gives localized
        eigenvalues that the short Lanczos run of the eigensolvers does not
        see (:func:`parsec_python.Eigensolvers.spectral_bounds.orbital_term_margin`).
        """
        if self.kinetic_potential is None:
            return 0.0
        cached = self.__dict__.get("_spectral_margin")
        if cached is None:
            from ..V_xc.meta_gga import GridGradient, KineticPotentialOperator

            cached = float(
                KineticPotentialOperator(
                    GridGradient(tuple(sp.csr_matrix(matrix) for matrix in self.gradient)),
                    self.kinetic_potential,
                    self.k_point,
                    gradient_flux=self.gradient_flux,
                ).spectral_margin
            )
            object.__setattr__(self, "_spectral_margin", cached)
        return cached

    def as_linear_operator(self) -> "LinearOperator":
        from scipy.sparse.linalg import LinearOperator

        operator = LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=complex,
        )
        if self.kinetic_potential is not None:
            # the eigensolvers' spectral bounds read it (orbital_term_margin)
            operator.spectral_margin = self.spectral_margin
        return operator


@dataclass(frozen=True)
class KPointExchangeHamiltonian:
    """``H_k + a_x V_ACE,k``: a Bloch Hamiltonian with the exact-exchange term of a hybrid functional.

    ``base`` is the :class:`KPointKohnShamHamiltonian` of the k-point and
    ``exchange`` its compressed exchange operator (an object with ``apply``
    and ``shape``, :class:`parsec_python.V_xc.bloch_exchange.BlochCompressedExchange`
    on the host), built from the Bloch factors of that k-point
    (:mod:`parsec_python.SCF.hybrid_periodic`); both act on ``u_nk``.
    """

    base: object
    exchange: object

    def __post_init__(self) -> None:
        if tuple(self.exchange.shape) != tuple(self.base.shape):
            raise ValueError("exact-exchange term does not match the Bloch Hamiltonian")

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    @property
    def k_point(self) -> np.ndarray:
        return self.base.k_point

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        block, was_1d = _as_block(vectors)
        result = self.base.apply(block) + self.exchange.apply(block)
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


__all__ = ["KPointExchangeHamiltonian", "KPointKohnShamHamiltonian"]
