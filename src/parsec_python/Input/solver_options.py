"""Eigensolver, SCF-control and start-vector labels of PARSEC 1.5 (input reader).

PARSEC 1.5 (``io/usrinputfile.F90`` of the 1.5 tree) reads a number of labels
that the earlier PARSEC did not have.  :func:`read_solver_options` reads those
of the eigensolvers, the SCF control and the random start vectors with 1.5's
validation and returns the fields of :class:`parsec_python.models.EigensolverSettings`
and :class:`parsec_python.models.SCFSettings` they set.  This port's defaults
were kept (2026-11-04): a label that is not written changes nothing, and
a label written explicitly gives PARSEC 1.5's behaviour where this port has it.

Labels that change the algorithm (all opt-in):

* ``Vector_Rng_Seed`` (1.5: -1, a seed from the clock; ``usrinputfile.F90:
  755-772``): the start vectors of the first solve come from the LAPACK seed
  it selects (:func:`parsec_python.Eigensolvers.lapack_random.lapack_seed_from_integer`)
  and the Lanczos starts from it; -1 written takes the seed from the clock as
  1.5 does and names it in a warning, so the run can be repeated.  1.5's
  numbers (gfortran ``random_number`` per MPI rank) are not reproduced.
* ``Smart_Subspace``, ``Chebyshev_Scaled``, ``Bounds_Method``, ``Bounds_Iter``
  (``usrinputfile.F90:2536-2544, 2737-2766``): PARSEC 1.5's SUBSPACE lower
  bound, scaled filter and Lanczos bounds on the readable eigensolvers.
  ``Bounds_Method: static`` leaves both bounds uninitialized in 1.5
  (``chebff.f90z:284-285``) and is refused.
* PARSEC.py extensions of this round, for the comparison of 1.5's rules:
  ``FF_Residual_Test`` (``off``, ``stop``: 1.5's residual-driven CHEBFF at
  ``Diag_Tolerance`` with its stop, ``continue``: the early exit only),
  ``Chebyshev_Degree_Decrease_After`` (4 = PARSEC 1.5's rule, the degree
  lowered from iteration 5 on, ``parsec.F90:1942-1952`` ``iter > 4``; 5 gives
  the earlier PARSEC's ``iter > 5``),
  ``Reorthogonalization_Threshold`` (0.1; 1.5 ``orth_normal.f90z:85`` 0.68) and
  ``Exact_State_Count`` (false; true: PARSEC 1.5's working space of exactly
  ``States_Num`` vectors, ``eigen_solver_module.F90:53``, instead of the buffer
  of ``Subspace_Buffer_Size``, which 1.5 does not read).  The parser reads the
  last one itself, beside ``Subspace_Buffer_Size``.

Labels with no effect here (a warning): ``Vector_Rng_Type`` and
``Old_Vector_Random`` (read and stored by 1.5, used only by code under
``#if 0``).  ``Thread_Random_Init``, ``Rritz_Logic``, ``Test_Logic`` and the
spectrum-slicing labels are read with the other tuning labels of 1.5 by the
parser itself (``_INPUT15_INERT_KEYS`` of :mod:`parsec_python.Input.parsec_input`).

``Filter_Restart_After``, ``Restart_Cheb_Remained_Times`` and
``Remained_Times_Atomic_Chrg_Den`` (``usrinputfile.F90:4600-4604``,
``parsec.F90:2944-3026``) restart the first solver every so many ionic steps;
see :func:`read_solver_options`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

FILTER_RESTART_KEYS = (
    "filter_restart_after",
    "restart_cheb_remained_times",
    "remained_times_atomic_chrg_den",
)
#: The labels :func:`read_solver_options` reads (lower case, as the parser keys them).
SOLVER_KEYS = frozenset(
    {
        "vector_rng_seed",
        "vector_rng_type",
        "old_vector_random",
        "smart_subspace",
        "chebyshev_scaled",
        "bounds_method",
        "bounds_iter",
        *FILTER_RESTART_KEYS,
        # PARSEC.py extensions of the 1.5 round.
        "ff_residual_test",
        "chebyshev_degree_decrease_after",
        "reorthogonalization_threshold",
        # read by the parser beside Subspace_Buffer_Size (it sets the buffer)
        "exact_state_count",
    }
)


@dataclass(frozen=True)
class SolverOptions:
    """The fields the labels of :data:`SOLVER_KEYS` set.

    ``eigensolver`` holds fields of :class:`parsec_python.models.EigensolverSettings`,
    ``scf`` fields of :class:`parsec_python.models.SCFSettings`; both are empty
    for an input that writes none of the labels.  ``filter_restart`` is
    ``(Filter_Restart_After, Restart_Cheb_Remained_Times,
    Remained_Times_Atomic_Chrg_Den)`` when the first is written, else ``None``.
    """

    eigensolver: dict = field(default_factory=dict)
    scf: dict = field(default_factory=dict)
    filter_restart: tuple[int, int, int] | None = None


def read_solver_options(
    scalar,
    one: Callable[..., str],
    optional_bool: Callable[..., bool],
    *,
    integer: Callable[..., int],
    real: Callable[..., float],
    eigensolver_method: str,
    filtering: bool,
    ionic_motion: bool,
    warnings: list[str],
    clock: Callable[[], float] = time.time,
) -> SolverOptions:
    """Read the PARSEC 1.5 labels of :data:`SOLVER_KEYS` (see the module text).

    ``integer``/``real`` are the parser's number readers (``_integer`` and
    ``_strict_float``), ``filtering`` whether later SCF steps run SUBSPACE
    (``solver%do_subsp``), ``ionic_motion`` whether the input moves atoms
    (relaxation or molecular dynamics).  ``clock`` gives the seed of
    ``Vector_Rng_Seed: -1`` (1.5 ``mysecond``, seconds of the wall clock).
    """

    # The parser defines the errors (it imports this module).
    from .parsec_input import ParsecInputError, UnsupportedParsecOptionError

    eigensolver: dict = {}
    scf: dict = {}

    def given(key: str) -> bool:
        return key in scalar

    # Start vectors (usrinputfile.F90:755-774, 2235-2236 of 1.5).
    if given("vector_rng_seed"):
        seed = integer(one("vector_rng_seed"), label="Vector_Rng_Seed")
        if seed == -1:
            seed = int(clock())
            warnings.append(
                f"Vector_Rng_Seed: -1 takes the seed from the clock, as PARSEC 1.5 does "
                f"('Vector Seed Randomized'): seed {seed} (write it to repeat the run)"
            )
        eigensolver["vector_seed"] = seed
        # The Lanczos and SUBSPACE streams of NumPy take a nonnegative seed.
        eigensolver["random_seed"] = seed % (1 << 32)
    inert = []
    if given("vector_rng_type"):
        integer(one("vector_rng_type"), label="Vector_Rng_Type")
        inert.append("Vector_Rng_Type")
    if given("old_vector_random"):
        optional_bool("old_vector_random")
        inert.append("Old_Vector_Random")
    if inert:
        warnings.append(
            f"{' and '.join(inert)}: PARSEC 1.5 reads and stores "
            f"{'it' if len(inert) == 1 else 'them'} and uses no such value in its vector "
            "routines (init_vector.f90z, code under #if 0); no effect here"
        )

    # SUBSPACE rules of PARSEC 1.5 (usrinputfile.F90:2536-2544, 2737-2766).
    if given("smart_subspace"):
        if filtering:
            eigensolver["subspace_lower_bound"] = (
                "smart" if optional_bool("smart_subspace") else "adaptive"
            )
        else:
            optional_bool("smart_subspace")
            warnings.append(
                "Smart_Subspace is read with the SUBSPACE filter only (PARSEC 1.5 reads it "
                "when solver%do_subsp); no effect here"
            )
    if given("chebyshev_scaled"):
        scaled = optional_bool("chebyshev_scaled")
        if filtering and scaled:
            eigensolver["scaled_subspace_filter"] = True
        elif filtering:
            warnings.append(
                "Chebyshev_Scaled: .false. is the filter of the earlier PARSEC, which "
                "PARSEC.py's SUBSPACE runs by default"
            )
    if given("bounds_method"):
        method = one("bounds_method").strip().lower()
        if method == "static":
            raise UnsupportedParsecOptionError(
                "Bounds_Method: static leaves the spectral bounds uninitialized in PARSEC 1.5 "
                "('nothing here yet', chebff.f90z:284-285); use lanczos or lanczos_full"
            )
        if method not in ("lanczos", "lanczos_full"):
            raise ParsecInputError(
                f"Bounds_Method: unknown value {method!r} (PARSEC 1.5 ierr 129); "
                "lanczos, lanczos_full or static"
            )
        eigensolver["lanczos_bounds"] = method
    if given("bounds_iter"):
        steps = integer(one("bounds_iter"), label="Bounds_Iter")
        if steps < 1:
            raise ParsecInputError("Bounds_Iter must be positive")
        eigensolver["lanczos_bound_steps"] = steps
        eigensolver.setdefault("lanczos_bounds", "lanczos")

    # PARSEC.py extensions of the 1.5 round.
    if given("ff_residual_test"):
        test = one("ff_residual_test").strip().lower()
        aliases = {"false": "off", "no": "off", "none": "off", "true": "stop", "parsec15": "stop"}
        test = aliases.get(test.strip(".").lower(), test)
        if test not in ("off", "stop", "continue"):
            raise ParsecInputError("FF_Residual_Test must be off, stop or continue")
        if test != "off":
            if eigensolver_method != "chebff":
                raise UnsupportedParsecOptionError(
                    f"FF_Residual_Test: {test} is PARSEC 1.5's residual-driven CHEBFF; "
                    f"Eigensolver={eigensolver_method} has no first filter"
                )
            eigensolver["first_filter_residual_test"] = test
    if given("chebyshev_degree_decrease_after"):
        after = integer(
            one("chebyshev_degree_decrease_after"), label="Chebyshev_Degree_Decrease_After"
        )
        if after < 0:
            raise ParsecInputError("Chebyshev_Degree_Decrease_After must be nonnegative")
        scf["degree_decrease_after"] = after
    if given("reorthogonalization_threshold"):
        threshold = real(
            one("reorthogonalization_threshold"), label="Reorthogonalization_Threshold"
        )
        if not 0.0 < threshold < 1.0:
            raise ParsecInputError("Reorthogonalization_Threshold must lie between 0 and 1")
        eigensolver["reorthogonalization_threshold"] = threshold

    # Restart of the first solver in ionic loops (usrinputfile.F90:4600-4604).
    filter_restart = None
    if any(given(key) for key in FILTER_RESTART_KEYS):
        after = integer(one("filter_restart_after", "5000"), label="Filter_Restart_After")
        times = integer(
            one("restart_cheb_remained_times", "3"), label="Restart_Cheb_Remained_Times"
        )
        density_times = integer(
            one("remained_times_atomic_chrg_den", "3"), label="Remained_Times_Atomic_Chrg_Den"
        )
        if after < 1:
            raise ParsecInputError(
                "Filter_Restart_After must be positive (PARSEC 1.5 takes mod(step, n))"
            )
        if not ionic_motion:
            warnings.append(
                "Filter_Restart_After, Restart_Cheb_Remained_Times and "
                "Remained_Times_Atomic_Chrg_Den are read for relaxation and molecular "
                "dynamics only; no effect here"
            )
        elif given("filter_restart_after") and filtering:
            filter_restart = (after, times, density_times)
    return SolverOptions(eigensolver=eigensolver, scf=scf, filter_restart=filter_restart)


__all__ = [
    "FILTER_RESTART_KEYS",
    "SOLVER_KEYS",
    "SolverOptions",
    "read_solver_options",
]
