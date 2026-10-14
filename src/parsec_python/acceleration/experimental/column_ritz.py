"""Complete experimental column-layout Ritz, including all three transposes."""
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from .mpi_domain import CollectiveDomainError


@dataclass(frozen=True)
class CompleteColumnRitzResult:
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    local_columns: Any
    residual_norms: np.ndarray | None
    overlap: np.ndarray
    projected_hamiltonian: np.ndarray
    condition_number: float
    orthogonality_error: float
    timings: dict


def _sum(layout, values):
    def buffers():
        packed = layout.xp.ascontiguousarray(values)
        if layout.transport == 'cuda':
            return packed, layout.xp.empty_like(packed)
        host = layout._host(packed)
        return host, np.empty_like(host)
    send, receive = layout._guard(buffers, 'Ritz reduction buffers')
    layout.comm.Allreduce(send, receive)
    return layout._host(receive)


def complete_column_ritz(layout, local_columns, *, compute_residuals=True,
                         condition_max=1.0e8):
    """H, X/HX transposes, generalized solve, row rotation and back transpose.

    Small matrices/coefficients are replicated; tall orbitals never gather.
    Timing values include local synchronization and collectives but are per
    rank; report their maximum across ranks.  An outer runner must abort on
    process/network failures, which Python collective guards cannot recover.
    """
    began = perf_counter()
    timings = {}
    def validate():
        if not np.isfinite(condition_max) or condition_max <= 1:
            raise ValueError('condition_max must be finite and exceed one')
        return layout._matrix(local_columns, layout.column_shape)
    x = layout._guard(validate, 'complete Ritz input')
    settings = layout.comm.allgather((bool(compute_residuals), float(condition_max)))
    if len(set(settings)) != 1:
        raise CollectiveDomainError('ranks disagree on complete Ritz settings')

    start = perf_counter()
    hx = layout._guard(lambda: layout.apply(x), 'column Hamiltonian action')
    timings['column_h_seconds'] = perf_counter()-start
    start = perf_counter()
    row_x = layout.columns_to_rows(x)
    row_hx = layout.columns_to_rows(hx)
    timings['two_forward_transposes_seconds'] = perf_counter()-start
    del hx
    start = perf_counter()
    partial = layout._guard(lambda: layout.xp.stack((row_x.T @ row_x, row_x.T @ row_hx)),
                            'local Gram and Hamiltonian projection')
    matrices = _sum(layout, partial)
    timings['projection_and_reduce_seconds'] = perf_counter()-start
    packet, error = None, None
    start = perf_counter()
    if layout.rank == 0:
        try:
            s = np.tril(matrices[0]) + np.tril(matrices[0], -1).T
            a = np.tril(matrices[1]) + np.tril(matrices[1], -1).T
            condition = float(np.linalg.cond(s))
            if not np.isfinite(condition) or condition > condition_max:
                raise np.linalg.LinAlgError(f'unsafe overlap condition {condition:.6g}')
            lower = np.linalg.cholesky(s)
            left = solve_triangular(lower, a, lower=True, check_finite=False)
            whitened = solve_triangular(lower, left.T, lower=True, check_finite=False).T
            whitened = np.tril(whitened) + np.tril(whitened, -1).T
            values, vectors = np.linalg.eigh(whitened)
            coefficients = solve_triangular(lower.T, vectors, lower=False, check_finite=False)
            ortho = float(np.max(np.abs(coefficients.T @ s @ coefficients-np.eye(len(values)))))
            if not np.isfinite(ortho) or ortho > 5.0e-10:
                raise np.linalg.LinAlgError(f'Ritz orthogonality audit failed: {ortho}')
            packet = (values, coefficients, s, a, condition, ortho)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = layout.comm.bcast(error, root=0)
    if error is not None:
        raise CollectiveDomainError('complete column Ritz root solve: '+error)
    values, coefficients, s, a, condition, ortho = layout.comm.bcast(packet, root=0)
    timings['root_solve_and_broadcast_seconds'] = perf_counter()-start
    start = perf_counter()
    device_coefficients = layout._guard(lambda: layout.xp.asarray(coefficients),
                                        'Ritz coefficient upload')
    rotated = layout._guard(lambda: row_x @ device_coefficients, 'row Ritz rotation')
    timings['row_rotation_seconds'] = perf_counter()-start
    residuals = None
    start = perf_counter()
    if compute_residuals:
        def squares():
            residual = row_hx @ device_coefficients - rotated*layout.xp.asarray(values)[None, :]
            return layout.xp.sum(residual*residual, axis=0)
        partial_norms = layout._guard(squares, 'row Ritz residuals')
        residuals = np.sqrt(np.maximum(_sum(layout, partial_norms), 0))
    timings['residuals_seconds'] = perf_counter()-start
    start = perf_counter()
    result = layout.rows_to_columns(rotated)
    timings['backward_transpose_seconds'] = perf_counter()-start
    layout._sync()
    timings['complete_ritz_seconds'] = perf_counter()-began
    return CompleteColumnRitzResult(values, coefficients, result, residuals, s, a,
                                    condition, ortho, timings)


__all__ = ['CompleteColumnRitzResult', 'complete_column_ritz']
