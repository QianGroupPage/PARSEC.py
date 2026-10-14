"""Correctness-first CuPy complex spinor Hamiltonian for self-consistent SOC.

This is a device-resident counterpart of
:class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`,
evaluating the same action

``H Psi = [T + diag(V_eff) + V_NL] Psi (per channel) + lzsz(Psi) + lsxy(Psi)``

entirely with CuPy arrays so it can sit inside a CuPy eigensolver without a
host round trip per Hamiltonian application.

Deliberately *not* the production real-valued path's architecture
(``acceleration.backends.cupy.CuPyHamiltonian``): that module's fused
multi-step Chebyshev-recurrence kernels, mixed-precision variant, and
symmetry-sector integration are hand-written ``extern "C" __global__`` CUDA C
operating on raw ``double*`` buffers, which cannot represent complex128
spinors without being rewritten from scratch in complex arithmetic. This
module instead reuses the *same* real kinetic/nonlocal sparse matrices the
production backend uploads, upcast once to complex128
(``cupyx.scipy.sparse`` natively supports complex dtypes, so no real/
imaginary splitting is needed), for the ordinary ``[T + V_eff + V_NL]``
per-channel term, and evaluates the spin-orbit ``L.S`` cross term
(``apply_lzsz``/``apply_lsxy``, ported line-for-line from
:mod:`parsec_python.Hamiltonian.spinor_operator` with ``numpy`` calls
replaced by ``cupy``) via CuPy's native dense linear algebra over each atom's
small support region -- a dense small-matrix contraction cuBLAS already
dispatches as a GPU kernel, not something a hand-written RawKernel would do
faster at these per-atom sizes (typically hundreds to a few thousand support
rows). Matching the real path's fused/mixed-precision/symmetry-sector
optimizations is an explicit follow-up once this baseline is validated on
real hardware -- there is no way to compile or execute CuPy/CUDA code in the
environment this module was written in.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers.perturbative_soc import (
    _LM_TO_L,
    _LM_TO_ML,
    AtomSpinOrbitProjectors,
)
from parsec_python.V_ion import NonlocalProjectorOperator

from .cupy import require_cupy


class _DeviceSOCProjector:
    """One SOC atom's ``v_ion``/``v_so`` pair uploaded to the device."""

    def __init__(self, cp, atom: AtomSpinOrbitProjectors) -> None:
        self.atom_index = atom.atom_index
        self.support_rows = cp.asarray(np.asarray(atom.support_rows, dtype=np.int64))
        self.v_ion = cp.asarray(np.asarray(atom.v_ion, dtype=np.complex128))
        self.v_so = cp.asarray(np.asarray(atom.v_so, dtype=np.complex128))
        self.sign = tuple(float(value) for value in atom.sign)


def _as_block(cp, vectors):
    array = cp.asarray(vectors, dtype=cp.complex128)
    if array.ndim == 1:
        return array[:, None], True
    return array, False


def _projector_overlaps(cp, projectors: Sequence[_DeviceSOCProjector], wavefunctions):
    """Device counterpart of
    :func:`parsec_python.Eigensolvers.perturbative_soc._projector_overlaps`."""

    dotio_by_atom = []
    dotso_by_atom = []
    for atom in projectors:
        block = wavefunctions[atom.support_rows, :]
        dotio = atom.v_ion.conj().T @ block
        dotso = atom.v_so.conj().T @ block
        sign = cp.where(
            cp.asarray(_LM_TO_L) == 1, atom.sign[0], atom.sign[1]
        )
        dotio_by_atom.append((dotio * sign[:, None]).T)
        dotso_by_atom.append((dotso * sign[:, None]).T)
    return dotio_by_atom, dotso_by_atom


def apply_lzsz(cp, projectors: Sequence[_DeviceSOCProjector], wavefunctions, spin_sign: int):
    """Device counterpart of
    :func:`parsec_python.Hamiltonian.spinor_operator.apply_lzsz`."""

    block, was_1d = _as_block(cp, wavefunctions)
    output = cp.zeros_like(block)
    if not projectors:
        return output[:, 0] if was_1d else output
    dotio_by_atom, dotso_by_atom = _projector_overlaps(cp, projectors, block)
    lm_to_ml = cp.asarray(_LM_TO_ML)
    lm_to_l = cp.asarray(_LM_TO_L)
    lz = spin_sign * 0.5 * lm_to_ml
    quarter_ll1 = 0.25 * lm_to_l * (lm_to_l + 1)
    for atom, dotio, dotso in zip(projectors, dotio_by_atom, dotso_by_atom):
        coeff_ion = (lz[None, :] * dotso).T
        coeff_so = (
            lz[None, :] * dotio - 0.5 * lz[None, :] * dotso + quarter_ll1[None, :] * dotso
        ).T
        contribution = atom.v_ion @ coeff_ion + atom.v_so @ coeff_so
        output[atom.support_rows, :] += contribution
    return output[:, 0] if was_1d else output


def apply_lsxy(cp, projectors: Sequence[_DeviceSOCProjector], wavefunctions_in, spin_sign: int):
    """Device counterpart of
    :func:`parsec_python.Hamiltonian.spinor_operator.apply_lsxy`."""

    block, was_1d = _as_block(cp, wavefunctions_in)
    output = cp.zeros_like(block)
    if not projectors:
        return output[:, 0] if was_1d else output
    dotio_by_atom, dotso_by_atom = _projector_overlaps(cp, projectors, block)
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
            weight = 0.5 * float(np.sqrt(float(number)))
            dotso_j = dotso[:, lm]
            dotio_j = dotio[:, lm]
            contribution = cp.outer(
                atom.v_ion[:, target], weight * dotso_j
            ) + cp.outer(atom.v_so[:, target], weight * (dotio_j - 0.5 * dotso_j))
            output[atom.support_rows, :] += contribution
    return output[:, 0] if was_1d else output


class CuPySpinorHamiltonian:
    """Device-resident complex spinor Hamiltonian (correctness-first baseline).

    ``negative_laplacian``/``effective_potential``/``nonlocal_operator`` are
    the same real, host-side pieces
    :class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`
    takes; ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output. Everything is uploaded once at construction; only
    ``effective_potential`` changes between SCF iterations (rebuild a new
    instance each iteration, matching the reference driver's own per-
    iteration Hamiltonian construction).
    """

    def __init__(
        self,
        negative_laplacian: sp.spmatrix,
        effective_potential: np.ndarray,
        nonlocal_operator: NonlocalProjectorOperator,
        soc_projectors: Sequence[AtomSpinOrbitProjectors],
    ) -> None:
        cp, cpsparse = require_cupy()
        self._cp = cp
        size = negative_laplacian.shape[0]
        if negative_laplacian.shape != (size, size):
            raise ValueError("negative_laplacian must be square")
        potential = np.asarray(effective_potential, dtype=np.float64)
        if potential.shape != (size,):
            raise ValueError("effective potential does not match the kinetic operator")
        if nonlocal_operator.shape != (size, size):
            raise ValueError("nonlocal operator does not match the kinetic operator")

        laplacian_complex = sp.csr_matrix(negative_laplacian, dtype=np.complex128)
        self._negative_laplacian = cpsparse.csr_matrix(laplacian_complex)
        self._effective_potential = cp.asarray(potential, dtype=cp.float64)
        projectors_complex = sp.csc_matrix(
            nonlocal_operator.projectors, dtype=np.complex128
        )
        self._projectors = cpsparse.csc_matrix(projectors_complex)
        self._projectors_h = cpsparse.csc_matrix(
            projectors_complex.conj().T.tocsc()
        )
        self._signs = cp.asarray(
            np.asarray(nonlocal_operator.signs, dtype=np.float64)
        )
        self._grid_size = size
        self._soc_projectors = tuple(
            _DeviceSOCProjector(cp, atom) for atom in soc_projectors
        )

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def shape(self) -> tuple[int, int]:
        return (2 * self._grid_size, 2 * self._grid_size)

    @property
    def dtype(self):
        return self._cp.complex128

    def _apply_channel(self, psi):
        return (
            self._negative_laplacian @ psi
            + self._effective_potential[:, None] * psi
            + self._projectors @ (self._signs[:, None] * (self._projectors_h @ psi))
        )

    def apply(self, spinor):
        cp = self._cp
        block = cp.asarray(spinor, dtype=cp.complex128)
        was_1d = block.ndim == 1
        if was_1d:
            block = block[:, None]
        n = self._grid_size
        if block.shape[0] != 2 * n:
            raise ValueError(
                f"spinor input must have 2*n_grid={2 * n} rows, got {block.shape[0]}"
            )
        psi_up = block[:n, :]
        psi_down = block[n:, :]

        q_up = self._apply_channel(psi_up)
        q_down = self._apply_channel(psi_down)

        if self._soc_projectors:
            q_up = q_up + apply_lzsz(cp, self._soc_projectors, psi_up, 1)
            q_up = q_up + apply_lsxy(cp, self._soc_projectors, psi_down, -1)
            q_down = q_down + apply_lzsz(cp, self._soc_projectors, psi_down, -1)
            q_down = q_down + apply_lsxy(cp, self._soc_projectors, psi_up, 1)

        result = cp.concatenate([q_up, q_down], axis=0)
        return result[:, 0] if was_1d else result

    def __matmul__(self, vectors):
        return self.apply(vectors)

    def as_eigensolver_operator(self) -> "CuPySpinorHamiltonian":
        """Identity hook matching the reference/accelerated scalar paths'
        ``getattr(hamiltonian, "as_eigensolver_operator", ...)`` convention
        (see ``SCF.self_consistent_soc.run_self_consistent_soc``). Unlike
        the scalar ``CuPyBoundHamiltonian``, there is no separate persistent
        device allocation to return here -- ``self`` already is the
        device-resident operator, exposing exactly the ``.shape``/``.apply``
        interface :mod:`chebff_spinor` needs.
        """

        return self


__all__ = ["CuPySpinorHamiltonian", "apply_lzsz", "apply_lsxy"]
