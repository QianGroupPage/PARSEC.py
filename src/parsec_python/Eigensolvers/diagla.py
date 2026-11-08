"""DIAGLA: K. Wu's block preconditioned Lanczos (Davidson) eigensolver PLESa.

Provenance
----------
A line-by-line translation of ``diagla_diag``/``zdiagla_diag`` in PARSEC's
``src/diagla.f90z`` ("PLESa -- block Preconditioned Lanczos Eigen-problem
Solver: the Abridged version", K. Wu, 1994-1995), of its preconditioner
``fdo5scl``/``zfdo5scl`` (``src/fdo5scl.f90z``) and of its small dense
eigensolver ``my_dsyev``/``my_zheev`` (``src/peig_inter.f90z``: LAPACK
``dsyev``/``zheev`` on the upper triangle).  PARSEC calls it from
``eigval.F90`` with the Krylov size ``kss = solver%kss0 = 20``, the window
``winsize = 5`` (``create_eigen_solver``, structures.F90:2433-2440) and the
tolerance ``solver%toler`` (``Diag_Tolerance``); ``evec`` holds the current
orbitals as initial guesses and has ``nn + winsize`` columns.

The algorithm (labels of the Fortran source):

* 10 (restart): stop when ``nec >= ned`` or the products are used up;
  otherwise copy the next ``lwin = min(winsize, kss/2)`` guesses
  ``evec(:, nec+1 : nec+lwin)`` into the basis ``bv`` and start with a block
  of ``lwin`` vectors, or of ``min(kss/2, jnd)`` when the previous basis was
  larger (old basis vectors beyond ``lwin`` are reused).
* 20 (Davidson step): classical Gram-Schmidt of each new basis vector
  against the locked eigenvectors and the basis, with Daniel's
  reorthogonalization test (``reorth = 0.25``, ``eps = 2.2e-16``, at most
  four passes, else ``info = -1``); a vector that vanishes is replaced by
  the old ``W`` column (from the second pass of products on) or by a unit
  vector at a pseudo-random grid row ``ird``.  ``W = A V``; the new columns
  of ``H = V^T W`` (upper triangle); ``H y = theta y``.  While the basis has
  room, only the first Ritz pair and its residual are formed (all ``lwin``
  pairs when the first one has converged); a full basis collapses to the
  ``lwin`` Ritz vectors followed by their residuals.
* Locking: pairs ``i = nec+1, ...`` (at most ``winsize - 1`` at a time) are
  accepted in turn while ``||r_i|| * ned < tol * i``, each moved down to
  its place in the ordered locked set; any acceptance restarts (label 10).
* Otherwise the residual of the first Ritz pair, preconditioned by
  ``fdo5scl``, is the next basis vector.  The shift the Fortran computes
  just before (the lower end of the trust region of the first Ritz value)
  is not used by ``fdo5scl``, and is not computed here.
* 90: ``info = 1`` when fewer than ``ned`` pairs converged within the
  budget; ``maxmvs`` becomes the products used.

``fdo5scl``: ``q_i = (p_i + (1/6) sum_{six first-shell neighbours} p_j)/2``
on the grid stencil (``parallel%neibs(1:6, :)`` in the order -x, +x, -y,
+y, -z, +z), a neighbour outside a cluster domain counting as zero;
spinor vectors (two grid blocks, up then down) are smoothed block by block.

A property of the Fortran preconditioner, kept: on a periodic grid with an
even number of points along every axis the alternating mode
``(-1)**(i+j+k)`` is an eigenvector of the six-neighbour sum with eigenvalue
-6, so ``fdo5scl`` maps it to zero.  Once the residual of the first
unlocked pair is dominated by that mode no correction can reduce it and
DIAGLA stalls until its budget is used (measured: H in a 6-bohr cube, 12
points per axis; 13 points converge in 430 products).  A cluster grid has
no such mode (its outside neighbours are zero).

The present Fortran input reader refuses ``Eigensolver: diagla``
(usrinputfile.F90:1416-1423, ``ierr = 126``: "DIAGLA eigensolver is to be
removed from PARSEC"); earlier PARSEC versions ran it
(``examples/tests/simple_cubic``).

Deviations: the Python port works on the full grid (PARSEC on the
symmetry-reduced wedge of one representation), so the rows ``ird`` of the
replacement unit vectors, which PARSEC draws from ``1..nwedge``, are drawn
from ``1..n_grid``; the neighbours are those of the full grid.  The
eigenvectors of the small problem come from SciPy's ``dsyev``/``zheev``
driver, like ``my_dsyev``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.linalg import LinAlgError, eigh

from .iterative import EIGVAL_MAXIMUM_CALLS, IterativeDiagonalizationError

# create_eigen_solver: solver%kss0 = 20, solver%winsize = 5 for DIAGLA.
DIAGLA_KRYLOV_SIZE = 20
DIAGLA_WINDOW_SIZE = 5
# diagla.f90z parameters.
DIAGLA_BLOCK_SIZE = 1
DIAGLA_REORTHOGONALIZATION = 0.25
DIAGLA_EPSILON = 2.2e-16

Preconditioner = Callable[[np.ndarray], np.ndarray]


def fdo5scl_preconditioner(neighbors: np.ndarray) -> Preconditioner:
    """Return ``fdo5scl``: half the vector plus half its first-shell mean.

    ``neighbors`` is the ``(n_grid, 6)`` row array of
    :func:`parsec_python.Grid.nearest_neighbor_rows` (``-1`` outside the
    domain).  The returned function maps ``(m * n_grid,)`` or
    ``(m * n_grid, k)`` arrays (``m`` grid blocks, 1 or 2) to arrays of the
    same shape.
    """

    rows = np.asarray(neighbors, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("neighbors must have shape (n_grid, 6)")
    grid_points = rows.shape[0]
    # Row n_grid of the padded block is PARSEC's zero entry nwedge + 1.
    padded_rows = np.where(rows < 0, grid_points, rows)
    one6th = 1.0 / 6.0

    def apply(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors)
        if vectors.shape[0] % grid_points:
            raise ValueError("vectors do not consist of whole grid blocks")
        out = np.empty_like(vectors)
        for start in range(0, vectors.shape[0], grid_points):
            block = vectors[start : start + grid_points]
            padded = np.concatenate([block, np.zeros_like(block[:1])])
            neighbor_sum = padded[padded_rows[:, 0]]
            for column in range(1, 6):
                neighbor_sum = neighbor_sum + padded[padded_rows[:, column]]
            out[start : start + grid_points] = 0.5 * (block + neighbor_sum * one6th)
        return out

    return apply


@dataclass(frozen=True)
class DiaglaSettings:
    """Controls of PARSEC's DIAGLA call.

    ``tolerance`` is ``Diag_Tolerance``; ``maximum_matvec`` is the budget
    (``solver%maxmv``) before the rule ``max(Maximum_matvec, n * n_states)``
    raises it; ``neighbors`` is the grid neighbour array the preconditioner
    needs (``None``: no preconditioner, the residual itself is the new
    basis vector -- for tests on matrices without a grid).
    """

    tolerance: float = 1.0e-4
    maximum_matvec: int = 30000
    krylov_size: int = DIAGLA_KRYLOV_SIZE
    window_size: int = DIAGLA_WINDOW_SIZE
    maximum_calls: int = EIGVAL_MAXIMUM_CALLS
    neighbors: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("DIAGLA tolerance must be positive")
        for name in ("maximum_matvec", "krylov_size", "window_size", "maximum_calls"):
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.krylov_size < 2 * DIAGLA_BLOCK_SIZE + 2:
            raise ValueError("krylov_size is too small for DIAGLA")


@dataclass(frozen=True)
class DiaglaResult:
    """Outcome of eigval.F90's DIAGLA loop.

    ``eigenvalues`` and ``vectors`` are the ``wanted`` lowest locked pairs
    (Ry; Euclidean-normalized columns); ``all_vectors`` is PARSEC's whole
    ``evec`` (``wanted + window`` columns), the initial guesses of the next
    solve.  ``residual_norms`` are the locked residuals ``||A x - theta x||``.
    """

    eigenvalues: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray
    all_vectors: np.ndarray
    converged_states: int
    matvecs: int
    calls: int
    info: int


@dataclass
class _Workspace:
    """The arrays and counters ``diagla_diag`` keeps between its labels."""

    evec: np.ndarray
    eval: np.ndarray
    res: np.ndarray
    bv: np.ndarray
    ww: np.ndarray
    hh: np.ndarray
    mv: int = 0
    ird: int = 0
    jnd: int = 0
    nec: int = 0


def _orthonormalize_column(
    work: _Workspace,
    i: int,
    *,
    nwedge: int,
) -> bool:
    """Label 30 for basis column ``i`` (1-based); False on ``info = -1``.

    Classical Gram-Schmidt against ``evec(:, 1:nec)`` and ``bv(:, 1:i-1)``,
    the coefficients kept in the scratch column ``ww(:, jnd+1)`` as in the
    Fortran, with Daniel's test for a repeated pass.
    """

    nec, jnd = work.nec, work.jnd
    j = i - 1
    column = i - 1
    scratch = jnd
    iw = 0
    s0 = 0.0
    while True:
        if iw > 4:
            return False
        iw += 1
        x = work.bv[:, column]
        if iw == 1:
            s0 = float(np.real(np.vdot(x, x)))
        locked = work.evec[:, :nec].conj().T @ x
        basis = work.bv[:, :j].conj().T @ x
        work.ww[:nec, scratch] = locked
        work.ww[nec : nec + j, scratch] = basis
        work.ww[nec + j, scratch] = s0
        x = x - work.evec[:, :nec] @ locked
        x = x - work.bv[:, :j] @ basis
        work.bv[:, column] = x
        if iw == 1:
            s0 = float(np.sqrt(np.real(work.ww[nec + j, scratch])))
        s1 = float(np.sqrt(np.real(np.vdot(x, x))))
        if s1 == 0.0:
            # The vector vanished: take what W holds, or a unit vector.
            if iw == 1 and work.mv > 0:
                work.bv[:, column] = work.ww[:, column]
                s1 = float(
                    np.sqrt(np.real(np.vdot(work.ww[:, column], work.ww[:, column])))
                )
            if s1 == 0.0:
                work.ird = (work.ird * 137 + i * 207 + 71) % nwedge + 1
                work.bv[work.ird - 1, column] = 1.0
                s0 = 1.0
            else:
                s0 = s1
            continue
        if s1 <= DIAGLA_REORTHOGONALIZATION * s0:
            # Daniel's test failed: orthogonalize once more.
            if s1 <= DIAGLA_EPSILON:
                work.ird += 1
                work.bv[:, column] *= 1.0 / s1
                if iw > 2:
                    work.ird = (work.ird * 391 + work.mv * 199 + 141) % nwedge + 1
                    work.bv[work.ird - 1, column] = 1.0
                    y = work.bv[:, column]
                    s0 = float(np.sqrt(np.real(np.vdot(y, y))))
                else:
                    s0 = 1.0
            else:
                s0 = s1
            continue
        scale = 1.0 / s1
        if s1 > 0.0 and scale > 0.0:
            work.bv[:, column] *= scale
            return True
        work.bv[:, column] = 0.0
        work.ird = (work.ird * 87 + 191) % nwedge + 1
        work.bv[work.ird - 1, column] = 1.0
        s0 = 1.0


def _projected_eigenpairs(hh: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """``my_dsyev('U', jpb, yy, kss, wk)``: eigenpairs of the upper triangle."""

    upper = np.triu(hh[:size, :size])
    return eigh(upper, lower=False, driver="ev", check_finite=False)


def diagla_diag(
    operator: Any,
    wanted_states: int,
    *,
    initial_vectors: np.ndarray | None = None,
    converged_states: int = 0,
    matvec_budget: int = 30000,
    tolerance: float = 1.0e-4,
    krylov_size: int = DIAGLA_KRYLOV_SIZE,
    window_size: int = DIAGLA_WINDOW_SIZE,
    preconditioner: Preconditioner | None = None,
    random_modulus: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """One ``diagla_diag`` call.

    Returns ``(eval, evec, res, nec, mv, info)`` with PARSEC's arrays of
    ``ned + winsiz`` entries/columns.  ``initial_vectors`` (up to
    ``ned + winsiz`` columns) are the guesses; missing ones are zero, as in
    a freshly allocated ``evec``.  ``random_modulus`` is ``nwedge`` of the
    replacement rows (the number of grid points; the operator dimension when
    omitted).
    """

    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    n = int(shape[0])
    ned = int(wanted_states)
    kss = int(krylov_size)
    winsiz = int(window_size)
    blksiz = DIAGLA_BLOCK_SIZE
    maxmvs = int(matvec_budget)
    nwedge = n if random_modulus is None else int(random_modulus)
    if not 1 <= ned < n:
        raise ValueError(f"DIAGLA needs between 1 and {n - 1} eigenpairs, got {ned}")
    if not 1 <= nwedge <= n:
        raise ValueError("random_modulus must be between 1 and the dimension")
    dtype = np.result_type(
        np.dtype(getattr(operator, "dtype", np.float64)),
        np.float64 if initial_vectors is None else np.asarray(initial_vectors).dtype,
    )
    if not np.issubdtype(dtype, np.complexfloating):
        dtype = np.dtype(np.float64)
    evec = np.zeros((n, ned + winsiz), dtype=dtype)
    if initial_vectors is not None:
        guesses = np.asarray(initial_vectors)
        if guesses.ndim != 2 or guesses.shape[0] != n:
            raise ValueError("initial_vectors must have one row per grid point")
        columns = min(guesses.shape[1], ned + winsiz)
        evec[:, :columns] = guesses[:, :columns]
    work = _Workspace(
        evec=evec,
        eval=np.zeros(ned + winsiz),
        res=np.zeros(ned + winsiz),
        bv=np.zeros((n, kss), dtype=dtype),
        ww=np.zeros((n, kss), dtype=dtype),
        hh=np.zeros((kss, kss), dtype=dtype),
        nec=int(converged_states),
    )
    info = 0
    lwin = min(winsiz, kss // 2)
    while True:
        # -- label 10: start/restart.
        if work.nec >= ned or work.mv >= maxmvs:
            break
        blk = lwin if work.jnd <= lwin else min(kss // 2, work.jnd)
        work.jnd = 0
        nec = work.nec
        work.bv[:, :lwin] = work.evec[:, nec : nec + lwin]
        while True:
            # -- label 20: the Davidson loop.
            jnd = work.jnd
            jpb = jnd + blk
            for i in range(jnd + 1, jpb + 1):
                if not _orthonormalize_column(work, i, nwedge=nwedge):
                    info = -1
                    break
            if info:
                break
            work.ww[:, jnd:jpb] = np.asarray(operator @ work.bv[:, jnd:jpb])
            work.mv += blk
            for i in range(jnd + 1, jpb + 1):
                work.hh[:i, i - 1] = work.bv[:, :i].conj().T @ work.ww[:, i - 1]
            try:
                wk, yy = _projected_eigenpairs(work.hh, jpb)
            except (LinAlgError, ValueError):
                info = -2
                break
            work.eval[nec : nec + lwin] = wk[:lwin]
            if jpb + blksiz <= kss and work.mv < maxmvs:
                # Only the first Ritz pair and its residual.
                work.evec[:, nec] = work.bv[:, :jpb] @ yy[:, 0]
                work.ww[:, jpb] = work.ww[:, :jpb] @ yy[:, 0]
                work.ww[:, jpb] -= wk[0] * work.evec[:, nec]
                work.res[nec] = np.sqrt(np.real(np.vdot(work.ww[:, jpb], work.ww[:, jpb])))
                if work.res[nec] * ned < tolerance * (nec + 1):
                    # The first pair converged: form all lwin Ritz pairs.
                    work.evec[:, nec + 1 : nec + lwin] = work.bv[:, :jpb] @ yy[:, 1:lwin]
                    work.bv[:, 1:lwin] = work.ww[:, :jpb] @ yy[:, 1:lwin]
                    for i in range(2, lwin + 1):
                        work.ww[:, i - 1] = (
                            work.bv[:, i - 1] - wk[i - 1] * work.evec[:, nec + i - 1]
                        )
                        work.res[nec + i - 1] = np.sqrt(
                            np.real(np.vdot(work.ww[:, i - 1], work.ww[:, i - 1]))
                        )
                work.jnd = jpb
            else:
                # Collapse the basis to the lwin Ritz vectors and append
                # their residuals.
                work.evec[:, nec : nec + lwin] = work.bv[:, :jpb] @ yy[:, :lwin]
                work.bv[:, :lwin] = work.ww[:, :jpb] @ yy[:, :lwin]
                for i in range(1, lwin + 1):
                    work.ww[:, i - 1] = work.bv[:, i - 1]
                    work.bv[:, i - 1] = work.evec[:, nec + i - 1]
                    work.ww[:, lwin + i - 1] = (
                        work.ww[:, i - 1] - wk[i - 1] * work.bv[:, i - 1]
                    )
                    work.res[nec + i - 1] = np.sqrt(
                        np.real(
                            np.vdot(work.ww[:, lwin + i - 1], work.ww[:, lwin + i - 1])
                        )
                    )
                for i in range(1, lwin + 1):
                    work.hh[:i, i - 1] = work.bv[:, :i].conj().T @ work.ww[:, i - 1]
                work.jnd = lwin
            blk = blksiz
            # Lock the converged pairs in the order of their eigenvalues.
            i = nec
            nconv = 0
            while i < nec + winsiz - 1:
                i += 1
                if not work.res[i - 1] * ned < tolerance * i:
                    break
                nconv += 1
                j = i
                while j > 1 and work.eval[j - 2] > work.eval[j - 1]:
                    work.eval[[j - 2, j - 1]] = work.eval[[j - 1, j - 2]]
                    work.res[[j - 2, j - 1]] = work.res[[j - 1, j - 2]]
                    work.evec[:, [j - 2, j - 1]] = work.evec[:, [j - 1, j - 2]]
                    j -= 1
            if nconv > 0 or work.mv >= maxmvs:
                work.nec = nec + nconv
                if jpb + blksiz > kss:
                    work.jnd = jpb
                break
            # The next basis vector: the preconditioned residual.
            start = work.jnd
            residual = work.ww[:, start : start + blk]
            work.bv[:, start : start + blk] = (
                residual if preconditioner is None else preconditioner(residual)
            )
        if info:
            break
    # -- label 90.
    if work.nec < ned and maxmvs <= work.mv and info >= 0:
        info = 1
    return work.eval, work.evec, work.res, work.nec, work.mv, info


def run_diagla(
    operator: Any,
    wanted_states: int,
    *,
    settings: DiaglaSettings = DiaglaSettings(),
    matvec_budget: int | None = None,
    initial_vectors: np.ndarray | None = None,
    random_modulus: int | None = None,
) -> DiaglaResult:
    """Run eigval.F90's DIAGLA loop for ``wanted_states`` eigenpairs.

    The call is repeated (keeping ``nec`` and the vectors, as eigval.F90
    passes them back in) until all pairs converged, at most
    ``settings.maximum_calls`` times; a call that reports ``info /= 0``
    ends the loop.  Raises :class:`IterativeDiagonalizationError`:
    ``ierr`` 401 for ``info`` -1 or -2, 402 when fewer than
    ``wanted_states`` pairs converged or the budget was used.
    """

    wanted_states = int(wanted_states)
    budget = int(settings.maximum_matvec if matvec_budget is None else matvec_budget)
    preconditioner = (
        None if settings.neighbors is None else fdo5scl_preconditioner(settings.neighbors)
    )
    if random_modulus is None and settings.neighbors is not None:
        random_modulus = int(np.shape(settings.neighbors)[0])
    vectors = initial_vectors
    nec = 0
    calls = 0
    info = 0
    matvecs = 0
    values = residuals = all_vectors = None
    # maxmvs is intent(inout): the first call gets the whole budget and
    # returns the products it used, which eigval.F90 hands to a repeated
    # call as its budget.  (A call ends with all pairs or with info /= 0,
    # so DIAGLA is never repeated in practice.)
    remaining = budget
    for _ in range(settings.maximum_calls):
        calls += 1
        values, all_vectors, residuals, nec, matvecs, info = diagla_diag(
            operator,
            wanted_states,
            initial_vectors=vectors,
            converged_states=nec,
            matvec_budget=remaining,
            tolerance=settings.tolerance,
            krylov_size=settings.krylov_size,
            window_size=settings.window_size,
            preconditioner=preconditioner,
            random_modulus=random_modulus,
        )
        vectors = all_vectors
        remaining = matvecs
        if info != 0:
            break
        if nec >= wanted_states:
            nec = wanted_states
            break
    # eigval.F90 tests info first (ierr 401; info = 1 is DIAGLA's own
    # "budget used up"), then the converged count and the budget (402).
    if info != 0 or nec < wanted_states or matvecs >= budget:
        raise IterativeDiagonalizationError(
            "DIAGLA",
            ierr=401 if info != 0 else 402,
            wanted=wanted_states,
            converged=nec,
            matvecs=matvecs,
            budget=budget,
            calls=calls,
            info=info,
        )
    return DiaglaResult(
        eigenvalues=np.array(values[:wanted_states]),
        vectors=np.array(all_vectors[:, :wanted_states]),
        residual_norms=np.array(residuals[:wanted_states]),
        all_vectors=np.array(all_vectors),
        converged_states=nec,
        matvecs=matvecs,
        calls=calls,
        info=info,
    )


__all__ = [
    "DIAGLA_BLOCK_SIZE",
    "DIAGLA_EPSILON",
    "DIAGLA_KRYLOV_SIZE",
    "DIAGLA_REORTHOGONALIZATION",
    "DIAGLA_WINDOW_SIZE",
    "DiaglaResult",
    "DiaglaSettings",
    "diagla_diag",
    "fdo5scl_preconditioner",
    "run_diagla",
]
