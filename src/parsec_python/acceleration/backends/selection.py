"""Resolve optional acceleration backends without silent substitutions.

``auto`` is the only mode allowed to fall back.  An explicitly requested
native or CuPy backend either runs or raises a useful error, which keeps
performance comparisons auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from parsec_python.models import SinglePointInput

from ..models import BackendName, BackendUnavailableError


FiniteDifferenceBuilder = Literal["reference", "native"]
HartreeBackend = Literal["scipy", "native", "cupy"]


@dataclass(frozen=True)
class BackendSelection:
    """Execution and component backends selected before preparation.

    ``selected`` names the Hamiltonian/eigensolver execution backend.
    Static finite-difference construction and the Hartree linear solve are
    recorded separately because ``auto`` may compose the native C++ setup
    kernels with the device-resident CuPy eigensolver.  Explicit selections
    remain single-backend paths for controlled comparisons.
    """

    requested: BackendName
    selected: str
    finite_difference_builder: FiniteDifferenceBuilder
    hartree_backend: HartreeBackend
    fallback_reasons: tuple[str, ...] = ()


def _native_status() -> tuple[bool, str | None]:
    try:
        from .native import native_available, native_unavailable_reason
    except (ImportError, OSError) as error:
        return False, f"native backend import failed: {error}"
    if native_available():
        return True, None
    return False, native_unavailable_reason() or "native extension is unavailable"


def _cupy_status() -> tuple[bool, str | None]:
    try:
        from .cupy import CuPyUnavailableError, require_cupy

        require_cupy()
    except (ImportError, OSError, CuPyUnavailableError) as error:
        return False, str(error)
    return True, None


def resolve_backend(
    requested: BackendName | str,
    problem: SinglePointInput,
) -> BackendSelection:
    """Resolve one backend using calculation capabilities and availability.

    Preference for ``auto`` is CuPy Hamiltonian/eigensolver execution combined
    with native C++ finite-difference construction and native Hartree CG when
    both optional runtimes are available.  If only one runtime is available,
    its complete supported path is used; SciPy is the final fallback.  The
    selected CHEBFF or CHEBDAV policy is preserved throughout: component
    selection never substitutes one eigensolver for the other.

    ``PARSEC_DEVICE`` (``--device`` of the command line) narrows ``auto``:
    ``cpu`` never probes or uses a GPU and takes the native path, or SciPy
    without the extension; ``gpu`` requires CuPy with a usable device and
    raises where there is none instead of running on the CPU.  An explicit
    backend that contradicts the device is refused.
    """

    from ..resources import device_policy

    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "scipy", "native", "cupy"}:
        raise ValueError(
            "backend must be one of 'auto', 'scipy', 'native', or 'cupy'"
        )
    device = device_policy()
    if device == "gpu" and normalized in {"scipy", "native"}:
        raise ValueError(
            f"backend {normalized!r} runs on the CPU, but a GPU run was requested "
            "(PARSEC_DEVICE=gpu or --device gpu)"
        )
    if device == "cpu" and normalized == "cupy":
        raise ValueError(
            "backend 'cupy' runs on the GPU, but a CPU run was requested "
            "(PARSEC_DEVICE=cpu or --device cpu)"
        )

    if normalized == "auto" and device == "cpu":
        # The GPU is left alone: CuPy is not imported and no CUDA context is
        # created.
        native_available, native_reason = _native_status()
        if native_available:
            return BackendSelection(
                requested="auto",
                selected="native",
                finite_difference_builder="native",
                hartree_backend="native",
            )
        return BackendSelection(
            requested="auto",
            selected="scipy",
            finite_difference_builder="reference",
            hartree_backend="scipy",
            fallback_reasons=(f"native skipped: {native_reason}",),
        )

    if normalized == "auto" and device == "gpu":
        cupy_available, cupy_reason = _cupy_status()
        if not cupy_available:
            raise BackendUnavailableError(
                "a GPU run was requested (--device gpu or PARSEC_DEVICE=gpu) "
                f"but no GPU is usable: {cupy_reason}"
            )

    if normalized == "scipy":
        return BackendSelection(
            requested="scipy",
            selected="scipy",
            finite_difference_builder="reference",
            hartree_backend="scipy",
        )

    if normalized == "cupy":
        available, reason = _cupy_status()
        if not available:
            raise BackendUnavailableError(
                f"CuPy backend was requested but is unavailable: {reason}"
            )
        return BackendSelection(
            requested="cupy",
            selected="cupy",
            finite_difference_builder="reference",
            hartree_backend="cupy",
        )

    if normalized == "native":
        available, reason = _native_status()
        if not available:
            raise BackendUnavailableError(
                f"native backend was requested but is unavailable: {reason}"
            )
        return BackendSelection(
            requested="native",
            selected="native",
            finite_difference_builder="native",
            hartree_backend="native",
        )

    # Probe both runtimes before deciding.  Native availability still matters
    # when CuPy is present because the fastest default is a component-aware
    # hybrid rather than the explicit, controlled pure-CuPy path.
    cupy_available, cupy_reason = _cupy_status()
    native_available, native_reason = _native_status()

    if cupy_available and native_available:
        return BackendSelection(
            requested="auto",
            selected="cupy",
            finite_difference_builder="native",
            hartree_backend="native",
        )

    if cupy_available:
        return BackendSelection(
            requested="auto",
            selected="cupy",
            finite_difference_builder="reference",
            hartree_backend="cupy",
            fallback_reasons=(
                f"native components skipped: {native_reason}",
            ),
        )

    if native_available:
        return BackendSelection(
            requested="auto",
            selected="native",
            finite_difference_builder="native",
            hartree_backend="native",
            fallback_reasons=(f"CuPy skipped: {cupy_reason}",),
        )

    return BackendSelection(
        requested="auto",
        selected="scipy",
        finite_difference_builder="reference",
        hartree_backend="scipy",
        fallback_reasons=(
            f"CuPy skipped: {cupy_reason}",
            f"native skipped: {native_reason}",
        ),
    )


__all__ = [
    "BackendSelection",
    "FiniteDifferenceBuilder",
    "HartreeBackend",
    "resolve_backend",
]
