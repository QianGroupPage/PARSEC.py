"""Lazy, exactly keyed finite-difference construction for accelerated runs."""

from .deferred import (
    DeferredNativeNegativeLaplacian,
    DeferredPeriodicNegativeLaplacian,
    PERIODIC_DEFER_ENV,
    materialize_negative_laplacian,
    periodic_laplacian_builder,
    periodic_negative_laplacian,
)

__all__ = [
    "DeferredNativeNegativeLaplacian",
    "DeferredPeriodicNegativeLaplacian",
    "PERIODIC_DEFER_ENV",
    "materialize_negative_laplacian",
    "periodic_laplacian_builder",
    "periodic_negative_laplacian",
]
