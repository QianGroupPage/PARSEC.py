"""Control flow PARSEC shares between its iterative eigensolvers.

Provenance
----------
``eigval.F90`` calls ARPACK (``arpk.f90z``) and DIAGLA (``diagla.f90z``)
from the same inner loop (lines 346-521): at most ``maxit = 5`` calls per
representation, leaving the loop when a call reports an error
(``info /= 0``) or when it converged all wanted eigenpairs
(``nec >= nn``).  After the loop (lines 562-600)

* a nonzero ``info`` stops PARSEC with ``ierr = 401``
  ("ERROR in eigval.F while working at representation ... error flag"),
* fewer converged pairs than wanted, or a last call that used
  ``solver%maxmv`` matrix-vector products or more, stops it with
  ``ierr = 402`` ("Iterative diagonalization has failed! ... Increase value
  of Maximum_matvec in parsec.in"), for every solver but CHEBDAV.

``solver%maxmv`` is ``Maximum_matvec`` (default 30000, usrinputfile.F90)
raised to ``ldn * nstate / nrep`` (parsec.F90:687), with ``ldn`` the rows of
one processor.  The Python port has one process and the full grid, so the
budget is ``max(Maximum_matvec, dimension * requested_states)`` with the
dimension of the eigenproblem.

Deviation: PARSEC also applies the ``ierr = 402`` test to CHEBFF and
SUBSPACE, whose matrix-vector counts the Python port does not record; the
test is applied to ARPACK and DIAGLA only.  CHEBFF and SUBSPACE do a fixed
polynomial work of about ``degree * cycles * (n_states + buffer)``
products, which reaches ``max(30000, n_grid * n_states)`` only for a
degree times cycle count of the order of the grid size and above 30000 /
(n_states + buffer); with the defaults (20 x 2 and 15) it cannot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator

# eigval.F90: integer, parameter :: maxit = 5
EIGVAL_MAXIMUM_CALLS = 5


class IterativeDiagonalizationError(RuntimeError):
    """PARSEC's eigval.F90 failure of an iterative eigensolver.

    ``ierr`` is 402 when too few eigenpairs converged or the
    matrix-vector budget was used up, 401 when the solver itself reported
    an error (``info`` is its flag: DIAGLA -1 orthogonalization failure,
    -2 failure of the small eigenproblem; ARPACK the ``_saupd`` code).
    """

    def __init__(
        self,
        solver: str,
        *,
        ierr: int,
        wanted: int,
        converged: int,
        matvecs: int,
        budget: int,
        calls: int,
        info: int = 0,
    ) -> None:
        self.solver = solver
        self.ierr = int(ierr)
        self.wanted = int(wanted)
        self.converged = int(converged)
        self.matvecs = int(matvecs)
        self.budget = int(budget)
        self.calls = int(calls)
        self.info = int(info)
        if self.ierr == 401:
            reason = {
                1: "used up its matrix-vector budget; increase Maximum_Matvec",
                -1: "failed to orthogonalize a basis vector",
                -2: "failed to solve the projected eigenproblem",
            }.get(self.info, "reported an error")
            message = (
                f"{solver} {reason} ({self.converged} of {self.wanted} "
                f"eigenpairs converged, {self.matvecs} matrix-vector "
                f"products; error flag {self.info}, eigval.F90 ierr 401)"
            )
        else:
            message = (
                f"Iterative diagonalization has failed! {solver} converged "
                f"{self.converged} of {self.wanted} eigenpairs with "
                f"{self.matvecs} matrix-vector products in the last of "
                f"{self.calls} call(s) (budget {self.budget}); increase "
                "Maximum_Matvec in parsec.in (eigval.F90 ierr 402)"
            )
        super().__init__(message)


def matvec_budget(maximum_matvec: int, dimension: int, requested_states: int) -> int:
    """``solver%maxmv = max(Maximum_matvec, ldn*nstate/nrep)`` (parsec.F90:687)."""

    return max(int(maximum_matvec), int(dimension) * int(requested_states))


class MatvecCounter:
    """Count the vectors an operator is applied to.

    ``as_linear_operator`` returns a SciPy operator whose every application
    adds the number of columns it receives to :attr:`count`, which is
    ARPACK's ``iparam(9)`` ("the number of OP*x") and DIAGLA's ``mv``.
    """

    def __init__(self, operator: Any) -> None:
        shape = getattr(operator, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("operator must expose a square two-dimensional shape")
        self.operator = operator
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(getattr(operator, "dtype", np.float64))
        self.count = 0

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Return ``H @ vectors`` for a vector or a block of columns."""

        vectors = np.asarray(vectors)
        self.count += 1 if vectors.ndim == 1 else int(vectors.shape[1])
        return np.asarray(self.operator @ vectors)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=self.dtype,
        )


__all__ = [
    "EIGVAL_MAXIMUM_CALLS",
    "IterativeDiagonalizationError",
    "MatvecCounter",
    "matvec_budget",
]
