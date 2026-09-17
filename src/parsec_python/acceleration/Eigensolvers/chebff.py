"""CuPy implementation of PARSEC's fixed-cycle first CHEBFF solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFCycle, ChebFFSettings
from ..backends.cupy import require_cupy
from .chebyshev import chebff_filter
from .lapack_random import LapackRandom
from .orthogonalize import orthonormalize_complete_subspace
from .rayleigh_ritz import DeviceRayleighRitzResult, rayleigh_ritz
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class DeviceChebFFState:
    """Complete buffered first-solve state; arrays remain on the GPU."""

    operator_dimension: int
    wanted_states: int
    eigenvalues: Any
    vectors: Any
    filter_lower_bound: float
    spectral_upper_bound: float
    smallest_ritz_value: float


@dataclass(frozen=True)
class DeviceChebFFResult:
    eigenvalues: Any
    vectors: Any
    state: DeviceChebFFState
    lanczos_bound: LanczosBoundResult
    cycles: tuple[ChebFFCycle, ...]
    last_rayleigh_ritz: DeviceRayleighRitzResult


def _initial_filter_lower_bound(lowest: float, upper: float) -> float:
    return (2.0 * lowest + upper) / 3.0


def _updated_filter_bounds(
    lower: float,
    upper: float,
    smallest_ritz: float,
    largest_ritz: float,
) -> tuple[float, float]:
    if largest_ritz >= upper:
        new_upper = largest_ritz + 0.5 * (largest_ritz - upper) + 1.0
        new_lower = min(lower, (3.0 * lower + new_upper) / 4.0)
        return float(new_lower), float(new_upper)
    new_lower = min(
        largest_ritz + 0.001 * (upper - smallest_ritz),
        largest_ritz + 0.05 * abs(largest_ritz),
    )
    return float(new_lower), float(upper)


def run_chebff(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebFFSettings = ChebFFSettings(),
    spectral_bound: LanczosBoundResult | None = None,
) -> DeviceChebFFResult:
    """Build the initial buffered eigensubspace without host round trips.

    The one intentional host-to-device transfer is the initial trial basis:
    it is generated with the reference :class:`LapackRandom`, preserving the
    PARSEC-compatible DLARNV stream, and then uploaded as float64.
    """

    cp, _ = require_cupy()
    dimension = int(operator.shape[0])
    if tuple(operator.shape) != (dimension, dimension):
        raise ValueError("operator must be square")
    wanted_states = int(wanted_states)
    if not 1 <= wanted_states <= dimension:
        raise ValueError("wanted_states is outside the operator dimension")

    basis_generator = LapackRandom()
    host_trial = basis_generator.uniform_minus_1_1(
        (dimension, wanted_states), column_major=True
    )
    vectors = cp.asarray(host_trial, dtype=cp.float64, order="F")
    # The persistent copy is the device basis.  Do not retain a second
    # N-by-state host allocation through all filter cycles.
    del host_trial

    if spectral_bound is None:
        bound_generator = np.random.default_rng(settings.random_seed)
        bound = lanczos_upper_bound(
            operator, steps=settings.lanczos_steps, rng=bound_generator
        )
    else:
        bound = spectral_bound
    upper_bound = float(bound.upper_bound)
    smallest_ritz = float(bound.lower_bound)
    lower_bound = _initial_filter_lower_bound(smallest_ritz, upper_bound)
    records: list[ChebFFCycle] = []
    last_ritz: DeviceRayleighRitzResult | None = None

    for cycle_number in range(1, settings.filter_cycles + 1):
        lower_in = lower_bound
        upper_in = upper_bound
        vectors = chebff_filter(
            operator,
            vectors,
            degree=settings.polynomial_degree,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            reference_eigenvalue=smallest_ritz,
            block_size=settings.block_size,
            reset_recurrence_per_block=settings.reset_recurrence_per_block,
        )
        # CHEBFF orthogonalizes the complete filtered trial space.  Route it
        # through the same size-adaptive implementation as later SUBSPACE
        # iterations: small problems retain PARSEC's selective MGS, moderate
        # GPU problems use one blocked QR, and multi-gigabyte tall bases use
        # memory-bounded TSQR.  All three construct the same FP64 column span.
        vectors = orthonormalize_complete_subspace(
            vectors, rng=basis_generator
        ).basis
        # CHEBFF uses only eigenvalues and rotated vectors.  Forming
        # ``(H Q) C - (Q C) Lambda`` here is both absent from PARSEC's CHEBFF
        # control flow and an unnecessary grid-by-state GPU operation.
        last_ritz = rayleigh_ritz(operator, vectors, compute_residuals=False)
        eigenvalues = last_ritz.eigenvalues
        vectors = last_ritz.wavefunctions
        if eigenvalues.shape != (wanted_states,):
            raise RuntimeError("Rayleigh--Ritz changed the working state count")
        smallest_ritz = float(eigenvalues[0].item())
        largest_ritz = float(eigenvalues[-1].item())
        lower_bound, upper_bound = _updated_filter_bounds(
            lower_bound,
            upper_bound,
            smallest_ritz,
            largest_ritz,
        )
        records.append(
            ChebFFCycle(
                number=cycle_number,
                lower_bound_in=float(lower_in),
                upper_bound_in=float(upper_in),
                lower_bound_out=float(lower_bound),
                upper_bound_out=float(upper_bound),
                smallest_ritz_value=smallest_ritz,
                largest_ritz_value=largest_ritz,
            )
        )

    assert last_ritz is not None
    state = DeviceChebFFState(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=last_ritz.eigenvalues,
        vectors=vectors,
        filter_lower_bound=lower_bound,
        spectral_upper_bound=upper_bound,
        smallest_ritz_value=smallest_ritz,
    )
    return DeviceChebFFResult(
        eigenvalues=last_ritz.eigenvalues,
        vectors=vectors,
        state=state,
        lanczos_bound=bound,
        cycles=tuple(records),
        last_rayleigh_ritz=last_ritz,
    )


__all__ = [
    "ChebFFCycle",
    "ChebFFSettings",
    "DeviceChebFFResult",
    "DeviceChebFFState",
    "run_chebff",
]
