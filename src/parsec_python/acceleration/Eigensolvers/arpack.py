"""ARPACK's role on the GPU: a device thick-restart Lanczos (``Eigensolver: arpack``).

Provenance
----------
PARSEC calls ARPACK from ``eigval.F90`` through ``arpk_diag`` (``arpk.f90z``,
built with ``-DUSEARPACK``); the reference port is
:mod:`parsec_python.Eigensolvers.arpack`, which drives SciPy's ARPACK
(``dsaupd``/``dseupd``) on the host.  This module runs the same method on the
device, for the full grid or one symmetry sector, with the parameters of
``arpk.f90z``:

* ``nev = nn`` wanted pairs (the working states), ``ncv = min(2*nev + 5, n)``
  Lanczos vectors, the smallest algebraic eigenvalues (``which = 'SA'``),
  exact shifts, at most ``180*nev`` restarts, the tolerance ``solver%toler``
  (``Diag_Tolerance``);
* the start vector of every call is ARPACK's ``dlarnv(idist=2)`` draw from the
  LAPACK seed ``dgetv0`` keeps (initially ``1, 3, 5, 7``), the same values as
  the reference (``info = 0`` on every call, eigval.F90:357-360);
* eigval.F90's loop: up to five calls until ``nec >= nn``, then ``ierr = 402``
  when the last call converged fewer pairs or used ``solver%maxmv`` products
  or more (:class:`parsec_python.Eigensolvers.iterative.IterativeDiagonalizationError`).

The restart is that of ARPACK's ``dsaup2`` written as a thick restart
(K. Wu and H. Simon, SIAM J. Matrix Anal. Appl. 22, 602 (2000): with exact
shifts the implicitly restarted Lanczos method keeps the span of the wanted
Ritz vectors, which a thick restart keeps explicitly):

* a factorization of ``ncv`` steps; every new vector is projected out of the
  basis by classical Gram-Schmidt with ARPACK's DGKS test (``dsaitr``: a
  second projection when the norm fell below ``0.717`` of its value before,
  at most two; a vector that is still in the span is replaced by a new
  random vector of the same LAPACK stream, as ``dgetv0`` does);
* the Ritz pairs of the projected matrix ``V^T A V``, their error bounds
  ``|beta_ncv * y_ncv,i|`` and ARPACK's convergence test (``dsconv``):
  ``bound_i <= tol * max(eps**(2/3), |theta_i|)`` for the ``nev`` smallest;
* a restart keeps ``nev + min(nconv, (ncv - nev)/2)`` Ritz vectors
  (``dsaup2``: "to prevent possible stagnation, adjust the number of Ritz
  values and the shifts"; ``nev = 1`` takes ``ncv/2`` or 2) and continues
  from the residual, which couples to every kept Ritz vector;
* the converged pairs are returned in ascending order (``dseupd`` and
  ``sortreal`` of arpk.f90z); a call that stops at its restart limit returns
  the converged ones, and the retry loop decides.

Deviations: the projected matrix is ``V^T A V`` as the projections compute
it (ARPACK keeps the tridiagonal ``alpha``/``beta`` of the same quantities);
a thick restart instead of the implicit QR steps of ``dsapps`` (the same
subspace in exact arithmetic, not the same rounding), so the Ritz values
agree with the reference to the tolerance and the product counts differ.
A sector whose dimension is ``nev`` (ARPACK requires ``nev < ncv <= n``)
is solved exactly by a dense eigensolver; the reference has no sectors.
Every sector keeps its own ``dlarnv`` stream from ``1, 3, 5, 7`` (PARSEC's
``dgetv0`` seed is one stream of the process across its representations),
so that the start vectors do not depend on which device finishes first:
the counts and pairs are the same on one and on several GPUs.  A
breakdown vector continues that stream (SciPy's ARPACK draws it from its
own seed).  Real symmetric operators only (the accelerated driver has no
complex Hamiltonian).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.arpack import ArpackCall, ArpackSettings
from parsec_python.Eigensolvers.iterative import (
    IterativeDiagonalizationError,
    matvec_budget,
)

from ..backends.cupy import require_cupy
from .lapack_random import LapackRandom
from .small_dense import symmetric_eigh

# dsaitr: a projected vector is accepted when its norm is above 0.717
# (about 1/sqrt(2)) of its norm before the projection.
DGKS_RATIO = 0.717
# dsconv: eps23 = dlamch('Epsilon-Machine')**(2/3).
EPS23 = float(np.finfo(np.float64).eps) ** (2.0 / 3.0)
# Rows of one tile of the products that form Ritz vectors (128 MiB a tile).
_ROTATION_TILE_BYTES = 1 << 27


@dataclass(frozen=True)
class DeviceArpackResult:
    """Outcome of eigval.F90's ARPACK loop on the device.

    ``eigenvalues`` (Ry, ascending, device) and ``vectors`` (device,
    ``(n, wanted)`` in Fortran order, orthonormal columns) are those of the
    last call; ``converged_states`` and ``matvecs`` its ``nec`` and
    ``maxmvs``; ``start_seed`` the LAPACK seed after it, which the next
    solve continues from; ``restarts`` the restarts of the last call.
    """

    eigenvalues: Any
    vectors: Any
    converged_states: int
    matvecs: int
    calls: tuple[ArpackCall, ...]
    start_seed: tuple[int, int, int, int]
    restarts: int = 0

    @property
    def total_matvecs(self) -> int:
        return int(sum(call.matvecs for call in self.calls))


def _apply(operator: Any, vector: Any):
    """``A v`` for one device vector, as a contiguous vector."""

    cp, _ = require_cupy()
    image = operator.apply(vector[:, None])
    return cp.ascontiguousarray(image.reshape(-1))


def _rotate_rows(cp, basis, coefficients, out) -> None:
    """``out = basis @ coefficients`` in row tiles; ``out`` may alias the leading columns of ``basis``.

    Every tile reads all columns of its rows before it writes, so the leading
    columns of ``basis`` can be overwritten in place (the thick restart).
    """

    rows = int(basis.shape[0])
    columns = max(1, int(coefficients.shape[1]))
    tile = max(1, min(rows, _ROTATION_TILE_BYTES // (8 * columns)))
    for start in range(0, rows, tile):
        stop = min(rows, start + tile)
        out[start:stop, :] = basis[start:stop, :] @ coefficients


def _random_orthogonal_vector(cp, random: LapackRandom, basis, dimension: int):
    """``dgetv0`` after a breakdown: a new draw of the stream, orthogonal to ``basis``."""

    vector = random.device_uniform_minus_1_1(dimension)
    for _ in range(2):
        if basis is not None and basis.shape[1]:
            vector -= basis @ (basis.T @ vector)
    norm = float(cp.linalg.norm(vector).item())
    if norm == 0.0:
        raise IterativeDiagonalizationError(
            "ARPACK", ierr=401, wanted=0, converged=0, matvecs=0, budget=0, calls=1, info=-9
        )
    return vector / norm


def _dense_solve(cp, operator, wanted: int, dimension: int):
    """The exact pairs of a tiny sector: ``A`` from ``dimension`` products."""

    identity = cp.eye(dimension, dtype=cp.float64, order="F")
    matrix = operator.apply(identity)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = cp.linalg.eigh(matrix)
    return (
        cp.ascontiguousarray(values[:wanted]),
        cp.asfortranarray(vectors[:, :wanted]),
        wanted,
        dimension,
        0,
    )


def device_arpack_diag(
    operator: Any,
    wanted_states: int,
    *,
    tolerance: float,
    maximum_restarts: int,
    random: LapackRandom,
):
    """One ``arpk_diag`` call on the device: ``(values, vectors, nec, matvecs, restarts)``.

    The start vector is the next draw of ``random`` (ARPACK's ``dgetv0``
    with ``info = 0``), which advances.  ``values`` ascending (device),
    ``vectors`` ``(n, nec_returned)`` in Fortran order; only the converged
    pairs are returned when the restart limit is reached.
    """

    cp, _ = require_cupy()
    dimension = int(operator.shape[0])
    wanted = int(wanted_states)
    if wanted < 1:
        raise ValueError("ARPACK needs at least one eigenpair")
    if wanted >= dimension:
        # dsaupd needs nev < ncv <= n; such a sector is tiny.
        if wanted > dimension:
            raise ValueError(
                f"ARPACK cannot return {wanted} eigenpairs of an operator of "
                f"dimension {dimension}"
            )
        return _dense_solve(cp, operator, wanted, dimension)
    tol = float(tolerance) if tolerance > 0 else float(np.finfo(np.float64).eps)
    basis_size = min(2 * wanted + 5, dimension)
    basis = cp.empty((dimension, basis_size), dtype=cp.float64, order="F")
    projected = np.zeros((basis_size, basis_size), dtype=np.float64)
    start = random.device_uniform_minus_1_1(dimension)
    norm = float(cp.linalg.norm(start).item())
    basis[:, 0] = start / norm
    del start
    matvecs = 0
    kept = 0
    residual = None
    beta = 0.0

    def extend(first: int) -> tuple[Any, float]:
        """Lanczos steps for columns ``first .. basis_size - 1``; the last residual and its norm."""

        nonlocal matvecs
        for column in range(first, basis_size):
            image = _apply(operator, basis[:, column])
            matvecs += 1
            active = basis[:, : column + 1]
            coefficients = active.T @ image
            image_norm = cp.sqrt(image @ image)
            image -= active @ coefficients
            image_rnorm = cp.sqrt(image @ image)
            host = cp.concatenate(
                [coefficients, image_norm[None], image_rnorm[None]]
            ).get()
            values = host[: column + 1]
            before, rnorm = float(host[-2]), float(host[-1])
            if rnorm <= DGKS_RATIO * before:
                # dsaitr steps 3-4: at most two further projections.
                accepted = False
                for _ in range(2):
                    correction = active.T @ image
                    image -= active @ correction
                    corrected = cp.sqrt(image @ image)
                    host = cp.concatenate([correction, corrected[None]]).get()
                    values = values + host[:-1]
                    previous, rnorm = rnorm, float(host[-1])
                    if rnorm > DGKS_RATIO * previous:
                        accepted = True
                        break
                if not accepted:
                    # The vector lies in the span: dsaitr sets the residual
                    # to zero and dgetv0 draws a new direction.
                    image.fill(0.0)
                    rnorm = 0.0
            projected[: column + 1, column] = values
            projected[column, : column + 1] = values
            if column + 1 == basis_size:
                return image, rnorm
            if rnorm == 0.0:
                basis[:, column + 1] = _random_orthogonal_vector(
                    cp, random, active, dimension
                )
                projected[column + 1, column] = projected[column, column + 1] = 0.0
            else:
                basis[:, column + 1] = image / rnorm
                projected[column + 1, column] = projected[column, column + 1] = rnorm
        raise AssertionError("the factorization ended without a residual")

    restarts = 0
    while True:
        residual, beta = extend(kept)
        values_device, vectors_device, theta = symmetric_eigh(
            projected, return_host_values=True
        )
        last_row = np.asarray(vectors_device[basis_size - 1, :].get(), dtype=np.float64)
        bounds = np.abs(beta * last_row)
        wanted_bounds = bounds[:wanted]
        wanted_values = theta[:wanted]
        converged_flags = wanted_bounds <= tol * np.maximum(EPS23, np.abs(wanted_values))
        converged = int(np.count_nonzero(converged_flags))
        if converged >= wanted or restarts >= maximum_restarts:
            break
        # dsaup2: keep nev + min(nconv, np/2) Ritz values (at least 2, or
        # ncv/2 for one wanted pair), then shift out the others.
        shifts = basis_size - wanted
        kept = wanted + min(converged, shifts // 2)
        if kept == 1 and basis_size >= 6:
            kept = basis_size // 2
        elif kept == 1 and basis_size > 2:
            kept = 2
        kept = min(kept, basis_size - 1)
        _rotate_rows(cp, basis, vectors_device[:, :kept], basis[:, :kept])
        projected.fill(0.0)
        projected[np.arange(kept), np.arange(kept)] = theta[:kept]
        coupling = beta * last_row[:kept]
        projected[kept, :kept] = coupling
        projected[:kept, kept] = coupling
        if beta == 0.0:
            basis[:, kept] = _random_orthogonal_vector(
                cp, random, basis[:, :kept], dimension
            )
            projected[kept, :kept] = 0.0
            projected[:kept, kept] = 0.0
        else:
            basis[:, kept] = residual / beta
        residual = None
        restarts += 1
    if converged >= wanted:
        selected = np.arange(wanted)
    else:
        selected = np.flatnonzero(converged_flags)
    coefficients = vectors_device[:, cp.asarray(selected)] if selected.size else None
    vectors = cp.empty((dimension, int(selected.size)), dtype=cp.float64, order="F")
    if selected.size:
        _rotate_rows(cp, basis, coefficients, vectors)
    eigenvalues = cp.asarray(theta[selected], dtype=cp.float64)
    del basis, residual
    return eigenvalues, vectors, converged, matvecs, restarts


def run_device_arpack(
    operator: Any,
    wanted_states: int,
    *,
    settings: ArpackSettings = ArpackSettings(),
    matvec_budget: int | None = None,
    start_seed: tuple[int, int, int, int] | None = None,
) -> DeviceArpackResult:
    """Run eigval.F90's ARPACK loop for ``wanted_states`` eigenpairs on the device.

    The counterpart of :func:`parsec_python.Eigensolvers.arpack.run_arpack`:
    ``matvec_budget`` is ``solver%maxmv`` (``settings.maximum_matvec`` when
    omitted); ``start_seed`` continues the LAPACK stream of an earlier
    solve.  Raises :class:`IterativeDiagonalizationError` with ``ierr`` 402
    when the last call converged fewer pairs or used the budget.
    """

    if getattr(operator, "_mpi_shared_group", None) is not None:
        raise NotImplementedError(
            "ARPACK on a sector basis that several MPI ranks share is not implemented"
        )
    wanted = int(wanted_states)
    budget = int(settings.maximum_matvec if matvec_budget is None else matvec_budget)
    random = LapackRandom(settings.start_seed if start_seed is None else start_seed)
    calls: list[ArpackCall] = []
    values = vectors = None
    converged = matvecs = restarts = 0
    for _ in range(settings.maximum_calls):
        values, vectors, converged, matvecs, restarts = device_arpack_diag(
            operator,
            wanted,
            tolerance=settings.tolerance,
            maximum_restarts=settings.restarts_per_state * wanted,
            random=random,
        )
        calls.append(ArpackCall(int(converged), int(matvecs)))
        if converged >= wanted:
            break
    if converged < wanted or matvecs >= budget:
        raise IterativeDiagonalizationError(
            "ARPACK",
            ierr=402,
            wanted=wanted,
            converged=converged,
            matvecs=matvecs,
            budget=budget,
            calls=len(calls),
        )
    return DeviceArpackResult(
        eigenvalues=values[:wanted],
        vectors=vectors[:, :wanted],
        converged_states=min(int(converged), wanted),
        matvecs=int(matvecs),
        calls=tuple(calls),
        start_seed=tuple(random.seed),
        restarts=int(restarts),
    )


__all__ = [
    "DeviceArpackResult",
    "device_arpack_diag",
    "matvec_budget",
    "run_device_arpack",
]
