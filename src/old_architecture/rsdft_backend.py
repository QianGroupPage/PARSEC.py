"""Backend-selection helpers for the refactored RSDFT driver.

The job of this file is to expose one function, ``select_solver_backend()``,
which returns a ``SolverBackend`` bundle pointing at either the CPU or GPU
implementations of the diagonalization and Hartree-solver routines.
"""

from __future__ import annotations

import os

import numpy as np

from .Eigensolvers.chefsi1 import chefsi1 as chefsi1_cpu
from .Eigensolvers.chsubsp import chsubsp as chsubsp_cpu
from .Eigensolvers.first_filt import first_filt as first_filt_cpu
from .Eigensolvers.lanczos import lanczos as lanczos_cpu
from .Eigensolvers.occupations import occupations as occupations_cpu
from .Eigensolvers.pcg import pcg as pcg_cpu
from .Mixer.mixer import mixer as mixer_cpu, reset_mixer as reset_mixer_cpu
from .V_ion.pseudoDiag import pseudoDiag as pseudo_diag_cpu
from .V_ion.pseudoNL_original import pseudoNL as pseudo_nl_cpu
from .V_xc.exc_nspn import exc_nspn as xc_cpu

from .rsdft_models import SolverBackend, SolverSettings

try:
    import cupy as cp
    from cupy._core import set_reduction_accelerators, set_routine_accelerators
except ImportError:
    cp = None
    set_reduction_accelerators = None
    set_routine_accelerators = None

try:
    from .Eigensolvers.chefsi1_gpu import chefsi1 as chefsi1_gpu
except ImportError:
    chefsi1_gpu = None

try:
    from .Eigensolvers.chsubsp_gpu import chsubsp as chsubsp_gpu
except ImportError:
    chsubsp_gpu = None

try:
    from .Eigensolvers.first_filt_gpu import first_filt as first_filt_gpu
except ImportError:
    first_filt_gpu = None

try:
    from .Eigensolvers.lanczos_gpu import lanczos as lanczos_gpu
except ImportError:
    lanczos_gpu = None

try:
    from .Eigensolvers.occupations_gpu import occupations as occupations_gpu
except ImportError:
    occupations_gpu = None

try:
    from .Eigensolvers.pcg_gpu import pcg as pcg_gpu
except ImportError:
    pcg_gpu = None

try:
    from .Mixer.msecant1_gpu import mixer_step as mixer_gpu, reset_mixer as reset_mixer_gpu
except ImportError:
    mixer_gpu = None
    reset_mixer_gpu = None

try:
    from .V_ion.pseudoDiag_gpu import pseudoDiag as pseudo_diag_gpu
except ImportError:
    pseudo_diag_gpu = None

try:
    from .V_ion.pseudoNL_original_gpu import pseudoNL as pseudo_nl_gpu
except ImportError:
    pseudo_nl_gpu = None

try:
    from .V_ion.pseudoDiag_cpp import pseudoDiag as pseudo_diag_native
except ImportError:
    pseudo_diag_native = None

try:
    from .V_ion.pseudoNL_original_cpp import pseudoNL as pseudo_nl_native
except ImportError:
    pseudo_nl_native = None

try:
    from .V_xc.exc_nspn_gpu import exc_nspn as xc_gpu, warmup as warmup_xc_gpu
except ImportError:
    xc_gpu = None
    warmup_xc_gpu = None

try:
    from .Laplacian.fd3d_gpu import warmup as warmup_fd3d_gpu
except ImportError:
    warmup_fd3d_gpu = None


def _env_choice(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off when set."
    )


def _select_ionic_setup(pseudo_diag_default, pseudo_nl_default, default_source: str):
    pseudo_diag_choice = _env_choice("PARSEC_NATIVE_PSEUDODIAG")
    pseudo_nl_choice = _env_choice("PARSEC_NATIVE_PSEUDONL")

    if pseudo_diag_choice is False:
        pseudo_diag_impl = pseudo_diag_default
        pseudo_diag_source = default_source
    elif pseudo_diag_native is not None:
        pseudo_diag_impl = pseudo_diag_native
        pseudo_diag_source = "cpp+openmp"
    elif pseudo_diag_choice is True:
        raise SystemExit(
            "PARSEC_NATIVE_PSEUDODIAG requested the native pseudoDiag wrapper, "
            "but it could not be imported. Rebuild the rsdft_native extension first."
        )
    else:
        pseudo_diag_impl = pseudo_diag_default
        pseudo_diag_source = default_source

    if pseudo_nl_choice is False:
        pseudo_nl_impl = pseudo_nl_default
        pseudo_nl_source = default_source
    elif pseudo_nl_native is not None:
        pseudo_nl_impl = pseudo_nl_native
        pseudo_nl_source = "cpp+openmp"
    elif pseudo_nl_choice is True:
        raise SystemExit(
            "PARSEC_NATIVE_PSEUDONL requested the native pseudoNL wrapper, "
            "but it could not be imported. Rebuild the rsdft_native extension first."
        )
    else:
        pseudo_nl_impl = pseudo_nl_default
        pseudo_nl_source = default_source

    return pseudo_diag_impl, pseudo_diag_source, pseudo_nl_impl, pseudo_nl_source


def configure_gpu_runtime() -> None:
    """Apply CuPy runtime settings needed by the GPU path."""
    if cp is None or set_reduction_accelerators is None or set_routine_accelerators is None:
        raise SystemExit("GPU backend requested but CuPy is not available.")
    set_reduction_accelerators([])
    set_routine_accelerators([])


def select_solver_backend(settings: SolverSettings) -> SolverBackend:
    """Return the CPU or GPU function bundle used by the solver.

    Input:
        settings: Normalized run settings. Only ``settings.use_gpu`` is used
            here, but the whole settings object is passed for clarity.

    Output:
        A ``SolverBackend`` dataclass containing the function references and
        array module for the selected execution path.
    """
    if settings.use_gpu:
        required_gpu_modules = {
            "cupy": cp,
            "first_filt_gpu": first_filt_gpu,
            "chefsi1_gpu": chefsi1_gpu,
            "lanczos_gpu": lanczos_gpu,
            "chsubsp_gpu": chsubsp_gpu,
            "occupations_gpu": occupations_gpu,
            "pcg_gpu": pcg_gpu,
            "exc_nspn_gpu": xc_gpu,
            "msecant1_gpu": mixer_gpu,
            "fd3d_gpu": warmup_fd3d_gpu,
        }
        missing = [name for name, impl in required_gpu_modules.items() if impl is None]
        if missing:
            raise SystemExit(
                "GPU backend requested but the following GPU modules could not be imported: "
                + ", ".join(missing)
            )

        configure_gpu_runtime()
        if warmup_xc_gpu is not None:
            warmup_xc_gpu()
        if warmup_fd3d_gpu is not None:
            warmup_fd3d_gpu()
        pseudo_diag_impl, pseudo_diag_source, pseudo_nl_impl, pseudo_nl_source = _select_ionic_setup(
            pseudo_diag_cpu,
            pseudo_nl_cpu,
            "cpu",
        )
        return SolverBackend(
            label="gpu",
            # The current ionic setup kernels remain dominated by CPU/Python
            # spline evaluation and host-side sparse assembly. Running those
            # stages on the GPU path adds transfer/kernel-launch overhead and
            # is slower than the native CPU implementations for known cases
            # such as H8C10.
            pseudo_diag=pseudo_diag_impl,
            pseudo_nl=pseudo_nl_impl,
            pseudo_diag_source=pseudo_diag_source,
            pseudo_nl_source=pseudo_nl_source,
            xc=xc_gpu,
            mixer=mixer_gpu,
            reset_mixer=reset_mixer_gpu,
            first_filt=first_filt_gpu,
            chefsi1=chefsi1_gpu,
            lanczos=lanczos_gpu,
            chsubsp=chsubsp_gpu,
            occupations=occupations_gpu,
            pcg=pcg_gpu,
            array_module=cp,
            cupy_module=cp,
        )

    pseudo_diag_impl, pseudo_diag_source, pseudo_nl_impl, pseudo_nl_source = _select_ionic_setup(
        pseudo_diag_cpu,
        pseudo_nl_cpu,
        "cpu",
    )
    return SolverBackend(
        label="cpu",
        pseudo_diag=pseudo_diag_impl,
        pseudo_nl=pseudo_nl_impl,
        pseudo_diag_source=pseudo_diag_source,
        pseudo_nl_source=pseudo_nl_source,
        xc=xc_cpu,
        mixer=mixer_cpu,
        reset_mixer=reset_mixer_cpu,
        first_filt=first_filt_cpu,
        chefsi1=chefsi1_cpu,
        lanczos=lanczos_cpu,
        chsubsp=chsubsp_cpu,
        occupations=occupations_cpu,
        pcg=pcg_cpu,
        array_module=np,
        cupy_module=cp,
    )
