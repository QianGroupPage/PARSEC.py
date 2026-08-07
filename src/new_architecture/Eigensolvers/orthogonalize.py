"""Selective two-pass Gram--Schmidt used by PARSEC eigensolvers.

Fortran provenance
------------------
PARSEC commit ``f432777750c2efc633adeb26faff641500b39b4a``,
``src/orth_normal.f90z``, routines ``orth_normal``/``zorth_normal``.

The implementation preserves the two PARSEC acceptance tests: reproject when
the first projected norm is at most ``0.1`` of the input norm, and replace a
numerically dependent vector when a second projection retains at most ``0.68``
of the once-projected norm.  MPI reductions are ordinary local inner products.

For a candidate column ``v`` and previously accepted basis ``Q``, the first
projection is

``v1 = v - Q*(Q.H@v)``.

If ``||v1|| <= 0.1*||v||``, PARSEC performs a second projection

``v2 = v1 - Q*(Q.H@v1)``.

It accepts ``v2`` only when ``||v2|| > 0.68*||v1||``.  Otherwise the candidate
is numerically dependent, is replaced by a LAPACK-stream random vector, and
the same tests restart.  This is the audited ``orth_normal`` policy; it should
not be confused with PARSEC's separate DGKS routine, whose ratio decisions
differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


_EPS = 2.22044605e-16
_EPS_BIG = 2.221e-16
_FIRST_REORTHOGONALIZATION_RATIO = 0.1
_SECOND_REORTHOGONALIZATION_RATIO = 0.68


class UniformRandom(Protocol):
    def uniform(
        self,
        low: float,
        high: float,
        size: int | tuple[int, ...],
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class OrthonormalizationResult:
    """Orthonormal basis and dependent-vector recovery diagnostics."""

    basis: np.ndarray
    random_replacements: int
    zero_replacements: int


def _replacement_vector(
    row_count: int,
    dtype: np.dtype,
    rng: UniformRandom,
) -> np.ndarray:
    if np.issubdtype(dtype, np.complexfloating):
        return (
            rng.uniform(-1.0, 1.0, row_count)
            + 1j * rng.uniform(-1.0, 1.0, row_count)
        ).astype(dtype, copy=False)
    # PARSEC uses random_array -> xLARNV(IDIST=2) in this recovery branch.
    return rng.uniform(-1.0, 1.0, row_count).astype(dtype, copy=False)


def orthonormalize(
    vectors: np.ndarray,
    *,
    existing_columns: int = 0,
    existing_are_orthonormal: bool = True,
    rng: UniformRandom | None = None,
    max_replacements: int = 100,
    copy: bool = True,
) -> OrthonormalizationResult:
    """Orthonormalize columns with PARSEC's ``orth_normal`` decisions.

    ``existing_columns`` corresponds to Fortran ``vsize``.  When
    ``existing_are_orthonormal`` is true, those columns are trusted and only
    later columns are processed.  When false, all columns are processed.
    Strict CHEBFF and the audited non-``BETA`` SUBSPACE path both call
    ``orth_normal`` with zero existing columns, so every filtered vector is
    processed against the columns accepted before it.  Random replacement
    prevents a collapsed filtered subspace from silently losing dimension.
    """

    input_array = np.asarray(vectors)
    result_dtype = np.dtype(np.result_type(input_array.dtype, np.float64))
    basis = np.asarray(vectors, dtype=result_dtype)
    if basis.ndim != 2:
        raise ValueError("vectors must be a two-dimensional column matrix")
    row_count, column_count = basis.shape
    if row_count < column_count:
        raise ValueError("cannot construct more orthonormal columns than rows")
    if not np.all(np.isfinite(basis)):
        raise ValueError("vectors must contain only finite values")
    if copy:
        basis = basis.copy()
    else:
        # CHEBDAV owns a large preallocated basis and appends only one small
        # block at a time.  Copying every already-orthonormal column on every
        # Davidson expansion would add O(N*k) memory traffic and a second
        # full basis allocation.  The opt-in in-place form follows the same
        # arithmetic below while updating that owned array directly.
        if basis is not input_array or input_array.dtype != result_dtype:
            raise ValueError(
                "copy=False requires a floating/complex array with no dtype conversion"
            )
        if not basis.flags.writeable:
            raise ValueError("copy=False requires a writeable array")

    existing_columns = int(existing_columns)
    if not 0 <= existing_columns <= column_count:
        raise ValueError("existing_columns must be between zero and the column count")
    max_replacements = int(max_replacements)
    if max_replacements < 0:
        raise ValueError("max_replacements cannot be negative")
    generator = np.random.default_rng() if rng is None else rng

    first_column = existing_columns if existing_are_orthonormal else 0
    random_replacements = 0
    zero_replacements = 0

    for column in range(first_column, column_count):
        # A random replacement returns here and is tested exactly like the
        # original candidate; no injected direction is trusted automatically.
        while True:
            vector = basis[:, column]
            input_norm = float(np.linalg.norm(vector))
            if input_norm < _EPS_BIG:
                if random_replacements >= max_replacements:
                    raise np.linalg.LinAlgError(
                        "orthogonalization exceeded the random replacement limit"
                    )
                basis[:, column] = _replacement_vector(
                    row_count, basis.dtype, generator
                )
                random_replacements += 1
                zero_replacements += 1
                continue

            previous = basis[:, :column]
            # First projection v1 = v-Q(Q^H v).
            if column:
                vector = vector - previous @ (previous.conj().T @ vector)
            projected_norm = float(np.linalg.norm(vector))

            if projected_norm > _FIRST_REORTHOGONALIZATION_RATIO * input_norm:
                if abs(projected_norm - 1.0) >= _EPS:
                    vector = vector / projected_norm
                basis[:, column] = vector
                break

            once_projected_norm = projected_norm
            # A large first-pass norm loss signals cancellation/dependence,
            # so repeat the projection before accepting this direction.
            if column:
                vector = vector - previous @ (previous.conj().T @ vector)
            projected_norm = float(np.linalg.norm(vector))

            if (
                projected_norm
                > _SECOND_REORTHOGONALIZATION_RATIO * once_projected_norm
            ):
                if abs(projected_norm - 1.0) >= _EPS:
                    vector = vector / projected_norm
                basis[:, column] = vector
                break

            if random_replacements >= max_replacements:
                raise np.linalg.LinAlgError(
                    "orthogonalization exceeded the random replacement limit"
                )
            # Two failed ratio tests mean the candidate lies numerically in
            # the existing span.  Inject a fresh direction and retry.
            basis[:, column] = _replacement_vector(
                row_count, basis.dtype, generator
            )
            random_replacements += 1

    return OrthonormalizationResult(
        basis=basis,
        random_replacements=random_replacements,
        zero_replacements=zero_replacements,
    )
