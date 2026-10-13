"""L-BFGS-B 3.0 bound-constrained minimizer, as PARSEC uses it for relaxation.

This module is a routine-by-routine translation of the limited-memory BFGS
code that PARSEC ships and calls from ``domove.f90``:

``lbfgsb.f`` (L-BFGS-B version 3.0, Zhu, Byrd, Lu, Nocedal and Morales)
    ``setulb``, ``mainlb``, ``active``, ``bmv``, ``cauchy``, ``cmprlb``,
    ``errclb``, ``formk``, ``formt``, ``freev``, ``hpsolb``, ``lnsrlb``,
    ``matupd``, ``projgr``, ``subsm``, ``dcsrch``, ``dcstep`` and the print
    routines ``prn1lb``, ``prn2lb``, ``prn3lb``;
``linpack.f``
    ``dpofa`` and ``dtrsl``;
``timer.f``
    ``timer``.

Reverse communication
---------------------
``setulb`` never evaluates the objective.  It returns with a task string and
the caller acts on it::

    state = LBFGSBState.start(n, m)              # move%cbfgs = 'START'
    f = setulb(state, x, lower, upper, nbd, f, g, factr, pgtol)
    while state.task[:2] == "FG":                # 'FG_START' or 'FG_LNSRCH'
        f = energy(x)                            # one SCF calculation in PARSEC
        g[:] = gradient(x)                       # minus the forces
        while True:
            f = setulb(state, x, lower, upper, nbd, f, g, factr, pgtol)
            if state.task[:5] != "NEW_X":        # 'NEW_X': call again
                break
    # state.task: 'CONVERGENCE: ...', 'ABNORMAL_TERMINATION_IN_LNSRCH'
    #             or 'ERROR: ...'

``domove.f90`` runs the inner loop once per ionic step: it leaves when the
task starts with ``FG`` (new coordinates are in ``x``) and ends the relaxation
for ``CONV``, ``WARN``, ``ERROR`` and ``ABNO``.  In its first step it makes the
``'START'`` call with the energy and forces of the initial coordinates and
then enters the loop with the same values.

``x`` and ``g`` are modified in place.  ``f`` is returned because Python
floats are immutable: the value is the input ``f`` except when the routine
restores the previous iterate (failed line search, ``STOP: CPU``), where it is
the function value of the restored point, as the Fortran sets ``f = fold``.

Persistent state
----------------
``LBFGSBState`` holds the arrays that the Fortran caller owns, with the same
lengths, the same partition and the same meaning of every slot, so that a
state written by Fortran PARSEC to ``relax_restart.dat`` can be continued here
and the reverse.  In ``domove.f90`` they are ``move%cbfgs(1:60)`` (task),
``move%cbfgs(61:120)`` (csave), ``move%lbfgs(4)`` (lsave), ``move%ibfgs(1:44)``
(isave), ``move%ibfgs(45:)`` (iwa), ``move%wbfgs`` (wa) and ``move%dbfgs(29)``
(dsave).

``wa`` (length ``2*m*n + 5*n + 11*m*m + 8*m``) is partitioned by the 1-based
offsets that ``setulb`` stores in ``isave(4:16)``; two-dimensional blocks are
stored column by column (Fortran order)::

    isave(4)  ws   (n, m)     S, the correction vectors s
    isave(5)  wy   (n, m)     Y, the correction vectors y
    isave(6)  sy   (m, m)     S'Y
    isave(7)  ss   (m, m)     S'S
    isave(8)  wt   (m, m)     Cholesky factor of theta*S'S + L D^-1 L'
    isave(9)  wn   (2m, 2m)   LEL' factorization of the subspace matrix K
    isave(10) snd  (2m, 2m)   lower triangle of N (inner products for wn)
    isave(11) z    (n)        Cauchy point, then subspace minimizer
    isave(12) r    (n)        reduced gradient; previous gradient in lnsrlb
    isave(13) d    (n)        search direction
    isave(14) t    (n)        breakpoints; previous x in lnsrlb
    isave(15) xp   (n)        safeguard copy for the projected Newton step
    isave(16) wa   (8m)       p, c, wbp, v of cauchy (2m each)

``iwa`` (length ``3*n``) is ``index(n)``, ``iwhere(n)``, ``indx2(n)``.  The
values stored in ``index`` and ``indx2`` are 1-based variable numbers and
``isave(27)`` (head), ``isave(29)`` (itail) and ``isave(40)`` (ileave) are
1-based positions, exactly as the Fortran stores them.

``isave(22:44)`` are the scalars of ``mainlb`` and ``dcsrch``: 22 nintol,
24 itfile, 25 iback, 26 nskip, 27 head, 28 col, 29 itail, 30 iter, 31 iupdat,
33 nseg, 34 nfgv, 35 info, 36 ifun, 37 iword, 38 nfree, 39 nact, 40 ileave,
41 nenter, 43 brackt (0/1), 44 stage.  ``dsave``: 1 theta, 2 fold, 3 tol,
4 dnorm, 5 epsmch, 6 cpu1, 7 cachyt, 8 sbtime, 9 lnscht, 10 time1, 11 gd,
12 stpmx, 13 sbgnrm, 14 stp, 15 gdold, 16 dtd, and 17:29 the line-search
scalars of ``dcsrch`` (ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin,
stmax, width, width1).  ``lsave``: prjctd, cnstnd, boxed, updatd.

Arithmetic
----------
Every floating-point expression is evaluated in the order of the Fortran
source.  The BLAS level-1 calls (``ddot``, ``daxpy``, ``dscal``, ``dcopy``)
follow the netlib reference BLAS: ``ddot`` is a strictly sequential sum.  With
an objective function that returns identical numbers, the iterates are then
bit-identical to the Fortran compiled without fused multiply-add and linked
to the reference BLAS.  A Fortran binary linked to an optimized BLAS (MKL,
OpenBLAS) or compiled with FMA differs from that reference, and from this
module, at rounding level.

Intentional deviations from the Fortran
---------------------------------------
* Nothing is written to files.  The Fortran writes its report on unit 9
  (``out.*`` in PARSEC; standard output in the original distribution) and the
  iteration table on unit 8 (``iterate.dat``).  The same text is handed to the
  optional ``output`` callback as ``output(unit, text)``.  PARSEC also opens
  ``bfgs.dat`` (unit 46) but never writes to it.
* The timing slots of ``dsave`` (6 to 10) stay zero unless a ``clock`` is
  supplied; they are never read by the algorithm.
* ``task`` and ``csave`` are Python strings without the Fortran blank padding;
  ``format_restart_block`` pads them to 60 columns.
* Division by zero and square roots of negative numbers give the IEEE results
  (infinity, NaN) of a compiled Fortran program instead of raising.  Fortran
  leaves MIN and MAX with a NaN operand to the processor; here every call
  site returns what the gfortran 13 code for x86-64 returns without
  optimization, so that a NaN or infinite ``f`` or ``g`` is handled as by that
  reference build.  Other builds of the Fortran may differ in that case.
* Two local variables of ``mainlb`` are undefined when ``prn3lb`` prints them
  in the abnormal-termination line of ``iterate.dat``: the CHARACTER*3
  ``word`` (not kept between calls; the gfortran build writes three NUL
  characters) and ``xstep`` when the line search returns before computing it
  (``info = -4``).  Here that line shows the status of the current iteration
  (always ``---``) and a zero step.  Nothing else depends on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "LBFGSBState",
    "UNIT_ITERATE",
    "UNIT_OUT",
    "format_restart_block",
    "list_directed",
    "parse_restart_block",
    "setulb",
]

UNIT_ITERATE = 8
"""Fortran unit of ``iterate.dat`` (``itfile`` in ``mainlb``)."""

UNIT_OUT = 9
"""Fortran unit of the L-BFGS-B report: ``out.*`` in PARSEC."""

OutputSink = Callable[[int, str], None]
Clock = Callable[[], float]

_EPSMCH = float(np.finfo(np.float64).eps)  # Fortran epsilon(1.0d0)
_TASK_LENGTH = 60

# Line-search constants of lnsrlb and dcsrch.
_BIG = 1.0e10
_FTOL = 1.0e-3
_GTOL = 0.9
_XTOL = 0.1
_P5 = 0.5
_P66 = 0.66
_XTRAPL = 1.1
_XTRAPU = 4.0


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


def _work_size(n: int, m: int) -> int:
    return 2 * m * n + 5 * n + 11 * m * m + 8 * m


@dataclass(eq=False)
class LBFGSBState:
    """Work arrays of ``setulb`` that persist between reverse-communication calls.

    Attributes
    ----------
    n, m:
        Number of variables and number of stored corrections (``move%mmax``,
        7 by default in PARSEC).
    task, csave:
        Fortran ``character*60`` strings without trailing blanks.  ``task`` is
        the job on entry and exit of ``setulb``; ``csave`` is the task of the
        line search ``dcsrch``.
    lsave:
        ``bool`` array of length 4 (prjctd, cnstnd, boxed, updatd).
    isave:
        ``int32`` array of length 44.
    dsave:
        ``float64`` array of length 29.
    wa:
        ``float64`` array of length ``2*m*n + 5*n + 11*m*m + 8*m``.
    iwa:
        ``int32`` array of length ``3*n``.

    The module docstring gives the layout of every array.
    """

    n: int
    m: int
    task: str
    csave: str
    lsave: np.ndarray
    isave: np.ndarray
    dsave: np.ndarray
    wa: np.ndarray
    iwa: np.ndarray

    def __post_init__(self) -> None:
        self.n = int(self.n)
        self.m = int(self.m)
        if self.n < 1 or self.m < 1:
            raise ValueError("n and m must be positive")
        self.task = str(self.task).rstrip(" ")[:_TASK_LENGTH]
        self.csave = str(self.csave).rstrip(" ")[:_TASK_LENGTH]
        self.lsave = np.ascontiguousarray(self.lsave, dtype=bool)
        self.isave = np.ascontiguousarray(self.isave, dtype=np.int32)
        self.dsave = np.ascontiguousarray(self.dsave, dtype=np.float64)
        self.wa = np.ascontiguousarray(self.wa, dtype=np.float64)
        self.iwa = np.ascontiguousarray(self.iwa, dtype=np.int32)
        expected = {
            "lsave": (self.lsave, 4),
            "isave": (self.isave, 44),
            "dsave": (self.dsave, 29),
            "wa": (self.wa, _work_size(self.n, self.m)),
            "iwa": (self.iwa, 3 * self.n),
        }
        for name, (array, length) in expected.items():
            if array.shape != (length,):
                raise ValueError(
                    f"{name} must have shape ({length},), got {array.shape}"
                )

    @classmethod
    def start(cls, n: int, m: int) -> "LBFGSBState":
        """Return a zero-filled state with ``task = 'START'``.

        This is the state of ``domove.f90`` after ``move%cbfgs = 'START'``;
        the Fortran leaves the freshly allocated arrays undefined, here they
        are zero.
        """
        n = int(n)
        m = int(m)
        if n < 1 or m < 1:
            raise ValueError("n and m must be positive")
        return cls(
            n=n,
            m=m,
            task="START",
            csave="",
            lsave=np.zeros(4, dtype=bool),
            isave=np.zeros(44, dtype=np.int32),
            dsave=np.zeros(29, dtype=np.float64),
            wa=np.zeros(_work_size(n, m), dtype=np.float64),
            iwa=np.zeros(3 * n, dtype=np.int32),
        )

    def copy(self) -> "LBFGSBState":
        """Return an independent copy of the state."""
        return LBFGSBState(
            n=self.n,
            m=self.m,
            task=self.task,
            csave=self.csave,
            lsave=self.lsave.copy(),
            isave=self.isave.copy(),
            dsave=self.dsave.copy(),
            wa=self.wa.copy(),
            iwa=self.iwa.copy(),
        )

    @property
    def iteration(self) -> int:
        """Number of the current iteration, ``isave(30)``."""
        return int(self.isave[29])

    @property
    def function_evaluations(self) -> int:
        """Total number of function and gradient evaluations, ``isave(34)``."""
        return int(self.isave[33])

    @property
    def projected_gradient_norm(self) -> float:
        """Infinity norm of the projected gradient, ``dsave(13)``."""
        return float(self.dsave[12])


# ---------------------------------------------------------------------------
# IEEE arithmetic helpers and reference BLAS level 1
# ---------------------------------------------------------------------------


def _div(a: float, b: float) -> float:
    """Return ``a/b`` with the IEEE result for a zero divisor."""
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0.0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _sqrt(a: float) -> float:
    """Return ``sqrt(a)`` with NaN for a negative argument."""
    if a < 0.0:
        return math.nan
    return math.sqrt(a)


def _fmax(a: float, b: float) -> float:
    """``max(a, b)`` that returns ``b`` when the operands are unordered or equal.

    Python's ``max(a, b)`` returns ``a`` in those cases.  The two forms differ
    only for a NaN operand or zeros of opposite sign, where Fortran leaves the
    result of MAX to the processor; each call site uses the form of the code
    that gfortran generates for it on x86-64 without optimization.
    """
    return a if a > b else b


def _fmin(a: float, b: float) -> float:
    """``min(a, b)`` that returns ``b`` when the operands are unordered or equal."""
    return a if a < b else b


def _ddot(a: np.ndarray, b: np.ndarray) -> float:
    """Reference-BLAS ``ddot``: sequential sum of products starting from +0."""
    if a.shape[0] == 0:
        return 0.0
    return float(np.add.accumulate(a * b)[-1]) + 0.0


def _ddot_list(count: int, a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for i in range(count):
        total = total + a[i] * b[i]
    return total


def _daxpy_list(count: int, da: float, dx: Sequence[float], dy: list) -> None:
    """Reference-BLAS ``daxpy`` on lists: ``dy = dy + da*dx``, no-op for da == 0."""
    if da == 0.0:
        return
    for i in range(count):
        dy[i] = dy[i] + da * dx[i]


# ---------------------------------------------------------------------------
# Fortran formatted output
# ---------------------------------------------------------------------------


def _fi(value: int, width: int) -> str:
    """Fortran ``Iw`` editing."""
    text = str(int(value))
    if len(text) > width:
        return "*" * width
    return text.rjust(width)


def _fexp(value: float, width: int, digits: int, letter: str) -> str:
    """Fortran ``1P,Dw.d`` / ``1P,Ew.d`` editing as gfortran writes it."""
    value = float(value)
    if value != value:
        text = "NaN"
    elif value == math.inf:
        text = "Infinity" if width >= 8 else "Inf"
    elif value == -math.inf:
        text = "-Infinity" if width >= 9 else "-Inf"
    else:
        mantissa, exponent = ("%.*E" % (digits, value)).split("E")
        number = int(exponent)
        sign = "+" if number >= 0 else "-"
        if abs(number) <= 99:
            text = "%s%s%s%02d" % (mantissa, letter, sign, abs(number))
        else:
            text = "%s%s%03d" % (mantissa, sign, abs(number))
    if len(text) > width:
        return "*" * width
    return text.rjust(width)


def _fd(value: float, width: int, digits: int) -> str:
    return _fexp(value, width, digits, "D")


def _fe(value: float, width: int, digits: int) -> str:
    return _fexp(value, width, digits, "E")


def _list_real(value: float) -> str:
    """REAL(8) field of gfortran list-directed output (25 columns)."""
    value = float(value)
    if value != value:
        return "NaN".rjust(25)
    if value == math.inf:
        return "Infinity".rjust(25)
    if value == -math.inf:
        return "-Infinity".rjust(25)
    mantissa, exponent = ("%.16E" % value).split("E")
    number = int(exponent)
    if value == 0.0 or -1 <= number <= 16:
        decimals = 16 if value == 0.0 else 16 - number
        text = "%.*f" % (decimals, value)
        if decimals == 0:
            text += "."
        return text.rjust(20) + "     "
    sign = "+" if number >= 0 else "-"
    return ("%sE%s%03d" % (mantissa, sign, abs(number))).rjust(25)


def list_directed(*items: object) -> str:
    """Format one record as gfortran's list-directed ``write(unit,*)`` does.

    ``str`` items are copied, ``bool`` gives ``T``/``F``, integers take 11
    columns and reals 25 columns with 17 significant digits; items are
    separated by one blank except two adjacent strings.  The record has no
    trailing newline.  ``domove.f90`` writes ``relax_restart.dat`` this way.
    """
    if not items:
        return ""
    pieces = [" "]
    previous_is_text = False
    for position, item in enumerate(items):
        is_text = isinstance(item, str)
        if position > 0 and not (is_text and previous_is_text):
            pieces.append(" ")
        if is_text:
            pieces.append(item)
        elif isinstance(item, (bool, np.bool_)):
            pieces.append("T" if item else "F")
        elif isinstance(item, (int, np.integer)):
            pieces.append(str(int(item)).rjust(11))
        elif isinstance(item, (float, np.floating)):
            pieces.append(_list_real(float(item)))
        else:
            raise TypeError(f"unsupported list-directed item: {item!r}")
        previous_is_text = is_text
    return "".join(pieces)


def _vector_block(label: str, values: Sequence[float]) -> str:
    """Format ``(/,a4,1p,6(1x,d11.4),/,(4x,1p,6(1x,d11.4)))``."""
    values = [float(v) for v in values]
    lines = ["", label.rjust(4) + "".join(" " + _fd(v, 11, 4) for v in values[:6])]
    if len(values) == 6:
        # The slash after the first group is executed before the format
        # runs out of data.
        lines.append("")
    for start in range(6, len(values), 6):
        chunk = values[start : start + 6]
        lines.append("    " + "".join(" " + _fd(v, 11, 4) for v in chunk))
    return "\n".join(lines) + "\n"


def _cauchy_block(values: Sequence[float]) -> str:
    """Format ``('Cauchy X =  ',/,(4x,1p,6(1x,d11.4)))``."""
    values = [float(v) for v in values]
    lines = ["Cauchy X =  "]
    for start in range(0, len(values), 6):
        chunk = values[start : start + 6]
        lines.append("    " + "".join(" " + _fd(v, 11, 4) for v in chunk))
    return "\n".join(lines) + "\n"


_REFRESH_SINGULAR = (
    "\n Singular triangular system detected;\n"
    "   refresh the lbfgs memory and restart the iteration.\n"
)
_REFRESH_FORMK = (
    "\n Nonpositive definiteness in Cholesky factorization in formk;\n"
    "   refresh the lbfgs memory and restart the iteration.\n"
)
_REFRESH_FORMT = (
    "\n Nonpositive definiteness in Cholesky factorization in formt;\n"
    "   refresh the lbfgs memory and restart the iteration.\n"
)
_REFRESH_LNSRCH = (
    "\n Bad direction in the line search;\n"
    "   refresh the lbfgs memory and restart the iteration.\n"
)

_INFO_MESSAGES = {
    -1: "\n Matrix in 1st Cholesky factorization in formk is not Pos. Def.\n",
    -2: "\n Matrix in 2st Cholesky factorization in formk is not Pos. Def.\n",
    -3: "\n Matrix in the Cholesky factorization in formt is not Pos. Def.\n",
    -4: (
        "\n Derivative >= 0, backtracking line search impossible.\n"
        "   Previous x, f and g restored.\n"
        " Possible causes: 1 error in function or gradient evaluation;\n"
        "                  2 rounding errors dominate computation.\n"
    ),
    -5: (
        "\n Warning:  more than 10 function and gradient\n"
        "   evaluations in the last line search.  Termination\n"
        "   may possibly be caused by a bad search direction.\n"
    ),
    -8: "\n The triangular system is singular.\n",
    -9: (
        "\n Line search cannot locate an adequate point after 20 function\n"
        "  and gradient evaluations.  Previous x, f and g restored.\n"
        " Possible causes: 1 error in function or gradient evaluation;\n"
        "                  2 rounding error dominate computation.\n"
    ),
}

_ITERATE_HEADER = (
    "RUNNING THE L-BFGS-B CODE\n"
    "\n"
    "it    = iteration number\n"
    "nf    = number of function evaluations\n"
    "nseg  = number of segments explored during the Cauchy search\n"
    "nact  = number of active bounds at the generalized Cauchy point\n"
    "sub   = manner in which the subspace minimization terminated:\n"
    "        con = converged, bnd = a bound was reached\n"
    "itls  = number of iterations performed in the line search\n"
    "stepl = step length used\n"
    "tstep = norm of the displacement (total step)\n"
    "projg = norm of the projected gradient\n"
    "f     = function value\n"
    "\n"
    "           * * *\n"
    "\n"
)

_SUMMARY_HEADER = (
    "\n"
    "           * * *\n"
    "\n"
    "Tit   = total number of iterations\n"
    "Tnf   = total number of function evaluations\n"
    "Tnint = total number of segments explored during Cauchy searches\n"
    "Skip  = number of BFGS updates skipped\n"
    "Nact  = number of active bounds at final generalized Cauchy point\n"
    "Projg = norm of the final projected gradient\n"
    "F     = final function value\n"
    "\n"
    "           * * *\n"
)


def _iterate_line(iter_: int, f: float, sbgnrm: float) -> str:
    """Format 1002 / 2001 of mainlb and prn2lb."""
    return "\nAt iterate%s    f= %s    |proj g|= %s\n" % (
        _fi(iter_, 5),
        _fd(f, 12, 5),
        _fd(sbgnrm, 12, 5),
    )


def _prn1lb(
    n: int,
    m: int,
    l: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    iprint: int,
    itfile: int,
    epsmch: float,
    out: OutputSink | None,
) -> None:
    """``prn1lb``: input data and headings."""
    if out is None or iprint < 0:
        return
    out(
        UNIT_OUT,
        "RUNNING THE L-BFGS-B CODE\n\n           * * *\n\n"
        "Machine precision =%s\n" % _fd(epsmch, 10, 3),
    )
    out(UNIT_OUT, list_directed("N = ", n, "    M = ", m) + "\n")
    if iprint >= 1:
        out(itfile, _ITERATE_HEADER + "Machine precision =%s\n" % _fd(epsmch, 10, 3))
        out(itfile, list_directed("N = ", n, "    M = ", m) + "\n")
        out(
            itfile,
            "\n   it   nf  nseg  nact  sub  itls  stepl    tstep     projg        f\n",
        )
        if iprint > 100:
            out(UNIT_OUT, _vector_block("L =", l))
            out(UNIT_OUT, _vector_block("X0 =", x))
            out(UNIT_OUT, _vector_block("U =", u))


def _subspace_word(iword: int) -> str:
    if iword == 0:
        return "con"
    if iword == 1:
        return "bnd"
    if iword == 5:
        return "TNT"
    return "---"


def _prn2lb(
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    iprint: int,
    itfile: int,
    iter_: int,
    nfgv: int,
    nact: int,
    sbgnrm: float,
    nseg: int,
    iword: int,
    iback: int,
    stp: float,
    xstep: float,
    out: OutputSink | None,
) -> None:
    """``prn2lb``: report after a successful line search."""
    if out is None:
        return
    word = _subspace_word(iword)
    if iprint >= 99:
        out(
            UNIT_OUT,
            list_directed("LINE SEARCH", iback, " times; norm of step = ", xstep)
            + "\n",
        )
        out(UNIT_OUT, _iterate_line(iter_, f, sbgnrm))
        if iprint > 100:
            out(UNIT_OUT, _vector_block("X =", x))
            out(UNIT_OUT, _vector_block("G =", g))
    elif iprint > 0:
        if iter_ % iprint == 0:
            out(UNIT_OUT, _iterate_line(iter_, f, sbgnrm))
    if iprint >= 1:
        out(
            itfile,
            " %s %s %s %s  %s %s  %s  %s %s %s\n"
            % (
                _fi(iter_, 4),
                _fi(nfgv, 4),
                _fi(nseg, 5),
                _fi(nact, 5),
                word,
                _fi(iback, 4),
                _fd(stp, 8, 1),
                _fd(xstep, 8, 1),
                _fd(sbgnrm, 10, 3),
                _fd(f, 10, 3),
            ),
        )


def _prn3lb(
    n: int,
    x: np.ndarray,
    f: float,
    task: str,
    iprint: int,
    info: int,
    itfile: int,
    iter_: int,
    nfgv: int,
    nintol: int,
    nskip: int,
    nact: int,
    sbgnrm: float,
    time: float,
    nseg: int,
    word: str,
    iback: int,
    stp: float,
    xstep: float,
    k: int,
    cachyt: float,
    sbtime: float,
    lnscht: float,
    out: OutputSink | None,
) -> None:
    """``prn3lb``: final report (convergence, abnormal exit or input error)."""
    if out is None or iprint < 0:
        return
    task_line = "\n%s\n" % task.ljust(_TASK_LENGTH)
    time_line = "\n Tot. User time%s seconds.\n\n" % _fe(time, 10, 3)
    if task[:5] != "ERROR":
        out(UNIT_OUT, _SUMMARY_HEADER)
        out(UNIT_OUT, "\n   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F\n")
        out(
            UNIT_OUT,
            "%s %s %s %s  %s %s  %s  %s\n"
            % (
                _fi(n, 5),
                _fi(iter_, 6),
                _fi(nfgv, 6),
                _fi(nintol, 6),
                _fi(nskip, 4),
                _fi(nact, 5),
                _fd(sbgnrm, 10, 3),
                _fd(f, 10, 3),
            ),
        )
        if iprint >= 100:
            out(UNIT_OUT, _vector_block("X =", x))
        if iprint >= 1:
            out(UNIT_OUT, list_directed(" F =", f) + "\n")
    out(UNIT_OUT, task_line)
    if info != 0:
        if info in (-1, -2, -3, -4, -5, -8, -9):
            out(UNIT_OUT, _INFO_MESSAGES[info])
        if info == -6:
            out(UNIT_OUT, list_directed(" Input nbd(", k, ") is invalid.") + "\n")
        if info == -7:
            out(
                UNIT_OUT,
                list_directed(" l(", k, ") > u(", k, ").  No feasible solution.")
                + "\n",
            )
    if iprint >= 1:
        out(
            UNIT_OUT,
            "\n Cauchy                time%s seconds.\n"
            " Subspace minimization time%s seconds.\n"
            " Line search           time%s seconds.\n"
            % (_fe(cachyt, 10, 3), _fe(sbtime, 10, 3), _fe(lnscht, 10, 3)),
        )
    out(UNIT_OUT, time_line)
    if iprint >= 1:
        if info == -4 or info == -9:
            out(
                itfile,
                " %s %s %s %s  %s %s  %s  %s      -          -\n"
                % (
                    _fi(iter_, 4),
                    _fi(nfgv, 4),
                    _fi(nseg, 5),
                    _fi(nact, 5),
                    word,
                    _fi(iback, 4),
                    _fd(stp, 8, 1),
                    _fd(xstep, 8, 1),
                ),
            )
        out(itfile, task_line)
        if info in (-1, -2, -3, -4, -5, -8, -9):
            out(itfile, _INFO_MESSAGES[info])
        out(itfile, time_line)


# ---------------------------------------------------------------------------
# LINPACK
# ---------------------------------------------------------------------------


def _dpofa(a: list, n: int, offset: int = 0) -> int:
    """LINPACK ``dpofa``: Cholesky factor of a symmetric positive definite matrix.

    ``a`` is a list of rows; the block of order ``n`` that starts at
    ``(offset, offset)`` is factored in place into ``R`` with ``A = R'R``
    (upper triangle).  Returns ``info``: 0, or the order of the leading minor
    that is not positive definite.
    """
    for j in range(n):
        column = offset + j
        s = 0.0
        for k in range(j):
            row = offset + k
            dot = 0.0
            for i in range(k):
                dot = dot + a[offset + i][row] * a[offset + i][column]
            t = a[row][column] - dot
            t = _div(t, a[row][row])
            a[row][column] = t
            s = s + t * t
        s = a[column][column] - s
        if s <= 0.0:
            return j + 1
        a[column][column] = _sqrt(s)
    return 0


def _dtrsl(t: list, n: int, b: list, offset: int, job: int) -> int:
    """LINPACK ``dtrsl``: solve a triangular system of order ``n``.

    ``t`` is a list of rows and the right-hand side is ``b[offset:offset+n]``,
    overwritten by the solution.  ``job`` is the Fortran argument: 00 solves
    ``T x = b`` with ``T`` lower triangular, 01 with ``T`` upper triangular,
    10 solves ``T' x = b`` with ``T`` lower and 11 with ``T`` upper triangular.
    Returns ``info``: 0, or the 1-based index of the first zero diagonal.
    """
    for j in range(n):
        if t[j][j] == 0.0:
            return j + 1
    case = 1
    if job % 10 != 0:
        case = 2
    if (job % 100) // 10 != 0:
        case = case + 2
    if case == 1:
        b[offset] = b[offset] / t[0][0]
        for j in range(1, n):
            temp = -b[offset + j - 1]
            if temp != 0.0:
                for i in range(j, n):
                    b[offset + i] = b[offset + i] + temp * t[i][j - 1]
            b[offset + j] = b[offset + j] / t[j][j]
    elif case == 2:
        b[offset + n - 1] = b[offset + n - 1] / t[n - 1][n - 1]
        for j in range(n - 2, -1, -1):
            temp = -b[offset + j + 1]
            if temp != 0.0:
                for i in range(j + 1):
                    b[offset + i] = b[offset + i] + temp * t[i][j + 1]
            b[offset + j] = b[offset + j] / t[j][j]
    elif case == 3:
        b[offset + n - 1] = b[offset + n - 1] / t[n - 1][n - 1]
        for j in range(n - 2, -1, -1):
            dot = 0.0
            for i in range(j + 1, n):
                dot = dot + t[i][j] * b[offset + i]
            b[offset + j] = b[offset + j] - dot
            b[offset + j] = b[offset + j] / t[j][j]
    else:
        b[offset] = b[offset] / t[0][0]
        for j in range(1, n):
            dot = 0.0
            for i in range(j):
                dot = dot + t[i][j] * b[offset + i]
            b[offset + j] = b[offset + j] - dot
            b[offset + j] = b[offset + j] / t[j][j]
    return 0


# ---------------------------------------------------------------------------
# L-BFGS-B subroutines
# ---------------------------------------------------------------------------


def _active(
    n: int,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    x: np.ndarray,
    iwhere: np.ndarray,
    iprint: int,
    out: OutputSink | None,
) -> tuple[bool, bool, bool]:
    """``active``: initialize ``iwhere`` and project ``x`` onto the feasible set.

    Returns ``(prjctd, cnstnd, boxed)``.
    """
    nbdd = 0
    prjctd = False
    cnstnd = False
    boxed = True
    xl = x.tolist()
    ll = l.tolist()
    ul = u.tolist()
    for i in range(n):
        if nbd[i] > 0:
            if nbd[i] <= 2 and xl[i] <= ll[i]:
                if xl[i] < ll[i]:
                    prjctd = True
                    xl[i] = ll[i]
                nbdd = nbdd + 1
            elif nbd[i] >= 2 and xl[i] >= ul[i]:
                if xl[i] > ul[i]:
                    prjctd = True
                    xl[i] = ul[i]
                nbdd = nbdd + 1
    if prjctd:
        x[:] = xl
    where = [0] * n
    for i in range(n):
        if nbd[i] != 2:
            boxed = False
        if nbd[i] == 0:
            where[i] = -1
        else:
            cnstnd = True
            if nbd[i] == 2 and ul[i] - ll[i] <= 0.0:
                where[i] = 3
            else:
                where[i] = 0
    iwhere[:] = where
    if out is not None:
        if iprint >= 0:
            if prjctd:
                out(
                    UNIT_OUT,
                    list_directed(
                        "The initial X is infeasible.  Restart with its projection."
                    )
                    + "\n",
                )
            if not cnstnd:
                out(UNIT_OUT, list_directed("This problem is unconstrained.") + "\n")
        if iprint > 0:
            out(
                UNIT_OUT,
                "\nAt X0 %s variables are exactly at the bounds\n" % _fi(nbdd, 9),
            )
    return prjctd, cnstnd, boxed


def _bmv(
    m: int,
    sy: list,
    wt: list,
    col: int,
    v: list,
    p: list,
    info: int,
) -> int:
    """``bmv``: product of the 2m x 2m middle matrix of the compact L-BFGS
    formula with the vector ``v[:2*col]``; the result is put in ``p[:2*col]``.

    ``sy`` and ``wt`` are lists of rows.  Returns ``info`` (unchanged for
    ``col == 0``, otherwise the ``info`` of ``dtrsl``).
    """
    if col == 0:
        return info
    p[col] = v[col]
    for i in range(1, col):
        i2 = col + i
        total = 0.0
        for k in range(i):
            total = total + _div(sy[i][k] * v[k], sy[k][k])
        p[i2] = v[i2] + total
    info = _dtrsl(wt, col, p, col, 11)
    if info != 0:
        return info
    for i in range(col):
        p[i] = _div(v[i], _sqrt(sy[i][i]))
    info = _dtrsl(wt, col, p, col, 1)
    if info != 0:
        return info
    for i in range(col):
        p[i] = _div(-p[i], _sqrt(sy[i][i]))
    for i in range(col):
        total = 0.0
        for k in range(i + 1, col):
            total = total + _div(sy[k][i] * p[col + k], sy[i][i])
        p[i] = p[i] + total
    return info


def _hpsolb(n: int, t: list, iorder: list, iheap: int) -> None:
    """``hpsolb``: move the least of ``t[:n]`` to ``t[n-1]``, heap the rest."""
    if iheap == 0:
        for k in range(2, n + 1):
            ddum = t[k - 1]
            indxin = iorder[k - 1]
            i = k
            while i > 1:
                j = i // 2
                if ddum < t[j - 1]:
                    t[i - 1] = t[j - 1]
                    iorder[i - 1] = iorder[j - 1]
                    i = j
                else:
                    break
            t[i - 1] = ddum
            iorder[i - 1] = indxin
    if n > 1:
        i = 1
        out = t[0]
        indxou = iorder[0]
        ddum = t[n - 1]
        indxin = iorder[n - 1]
        while True:
            j = i + i
            if j <= n - 1:
                if t[j] < t[j - 1]:
                    j = j + 1
                if t[j - 1] < ddum:
                    t[i - 1] = t[j - 1]
                    iorder[i - 1] = iorder[j - 1]
                    i = j
                    continue
            break
        t[i - 1] = ddum
        iorder[i - 1] = indxin
        t[n - 1] = out
        iorder[n - 1] = indxou


def _cauchy(
    n: int,
    x: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    g: np.ndarray,
    iorder: np.ndarray,
    iwhere: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    xcp: np.ndarray,
    m: int,
    wy: np.ndarray,
    ws: np.ndarray,
    sy: np.ndarray,
    wt: np.ndarray,
    theta: float,
    col: int,
    head: int,
    wa: np.ndarray,
    nseg: int,
    iprint: int,
    sbgnrm: float,
    info: int,
    epsmch: float,
    out: OutputSink | None,
) -> tuple[int, int]:
    """``cauchy``: generalized Cauchy point along the projected gradient.

    ``wa`` is the 8m work array; its four quarters are ``p``, ``c``, ``wbp``
    and ``v``.  ``iorder`` is ``indx2`` of ``mainlb`` and ``t`` receives the
    breakpoints.  Returns ``(nseg, info)``; ``nseg`` is unchanged when the
    routine returns before any segment is examined.
    """
    verbose = out is not None
    if sbgnrm <= 0.0:
        if verbose and iprint >= 0:
            out(UNIT_OUT, list_directed("Subgnorm = 0.  GCP = X.") + "\n")
        xcp[:] = x
        return nseg, info

    m2 = 2 * m
    xl = x.tolist()
    ll = l.tolist()
    ul = u.tolist()
    gl = g.tolist()
    where = iwhere.tolist()
    order = iorder.tolist()
    tl_ = t.tolist()
    dl = d.tolist()
    work = wa.tolist()
    p = work[0:m2]
    c = work[m2 : 2 * m2]
    wbp = work[2 * m2 : 3 * m2]
    v = work[3 * m2 : 4 * m2]
    xcpl = xcp.tolist()
    if col > 0:
        wyl = wy.tolist()
        wsl = ws.tolist()
        syl = sy.tolist()
        wtl = wt.tolist()
    else:
        wyl = wsl = syl = wtl = None

    def store() -> None:
        iwhere[:] = where
        iorder[:] = order
        t[:] = tl_
        d[:] = dl
        xcp[:] = xcpl
        wa[0:m2] = p
        wa[m2 : 2 * m2] = c
        wa[2 * m2 : 3 * m2] = wbp
        wa[3 * m2 : 4 * m2] = v

    bnded = True
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    f1 = 0.0
    if verbose and iprint >= 99:
        out(UNIT_OUT, "\n---------------- CAUCHY entered-------------------\n")

    for i in range(col2):
        p[i] = 0.0

    # Bound status and breakpoint of every variable; p = W'd is built up.
    tl = 0.0
    tu = 0.0
    for i in range(n):
        neggi = -gl[i]
        nb = nbd[i]
        if where[i] != 3 and where[i] != -1:
            if nb <= 2:
                tl = xl[i] - ll[i]
            if nb >= 2:
                tu = ul[i] - xl[i]
            xlower = nb <= 2 and tl <= 0.0
            xupper = nb >= 2 and tu <= 0.0
            where[i] = 0
            if xlower:
                if neggi <= 0.0:
                    where[i] = 1
            elif xupper:
                if neggi >= 0.0:
                    where[i] = 2
            else:
                if abs(neggi) <= 0.0:
                    where[i] = -3
        pointr = head
        if where[i] != 0 and where[i] != -1:
            dl[i] = 0.0
        else:
            dl[i] = neggi
            f1 = f1 - neggi * neggi
            for j in range(col):
                p[j] = p[j] + wyl[i][pointr - 1] * neggi
                p[col + j] = p[col + j] + wsl[i][pointr - 1] * neggi
                pointr = pointr % m + 1
            if nb <= 2 and nb != 0 and neggi < 0.0:
                nbreak = nbreak + 1
                order[nbreak - 1] = i + 1
                tl_[nbreak - 1] = _div(tl, -neggi)
                if nbreak == 1 or tl_[nbreak - 1] < bkmin:
                    bkmin = tl_[nbreak - 1]
                    ibkmin = nbreak
            elif nb >= 2 and neggi > 0.0:
                nbreak = nbreak + 1
                order[nbreak - 1] = i + 1
                tl_[nbreak - 1] = _div(tu, neggi)
                if nbreak == 1 or tl_[nbreak - 1] < bkmin:
                    bkmin = tl_[nbreak - 1]
                    ibkmin = nbreak
            else:
                nfree = nfree - 1
                order[nfree - 1] = i + 1
                if abs(neggi) > 0.0:
                    bnded = False

    if theta != 1.0:
        for j in range(col, col2):
            p[j] = theta * p[j]

    xcpl = xl[:]

    if nbreak == 0 and nfree == n + 1:
        store()
        if verbose and iprint > 100:
            out(UNIT_OUT, _cauchy_block(xcpl))
        return nseg, info

    for j in range(col2):
        c[j] = 0.0

    f2 = -theta * f1
    f2_org = f2
    if col > 0:
        info = _bmv(m, syl, wtl, col, p, v, info)
        if info != 0:
            store()
            return nseg, info
        f2 = f2 - _ddot_list(col2, v, p)
    dtm = _div(-f1, f2)
    tsum = 0.0
    nseg = 1
    if verbose and iprint >= 99:
        out(UNIT_OUT, list_directed("There are ", nbreak, "  breakpoints ") + "\n")

    all_fixed = False
    if nbreak != 0:
        nleft = nbreak
        iter_ = 1
        tj = 0.0
        while True:
            # Find the next smallest breakpoint.
            tj0 = tj
            if iter_ == 1:
                tj = bkmin
                ibp = order[ibkmin - 1]
            else:
                if iter_ == 2:
                    if ibkmin != nbreak:
                        tl_[ibkmin - 1] = tl_[nbreak - 1]
                        order[ibkmin - 1] = order[nbreak - 1]
                _hpsolb(nleft, tl_, order, iter_ - 2)
                tj = tl_[nleft - 1]
                ibp = order[nleft - 1]

            dt = tj - tj0

            if verbose and dt != 0.0 and iprint >= 100:
                out(
                    UNIT_OUT,
                    "\nPiece    %s --f1, f2 at start point  %s %s\n"
                    % (_fi(nseg, 3), _fd(f1, 11, 4), _fd(f2, 11, 4)),
                )
                out(
                    UNIT_OUT,
                    "Distance to the next break point =  %s\n" % _fd(dt, 11, 4),
                )
                out(
                    UNIT_OUT,
                    "Distance to the stationary point =  %s\n" % _fd(dtm, 11, 4),
                )

            # A minimizer within this interval: locate the GCP and return.
            if dtm < dt:
                break

            # Otherwise fix one variable and zero its component of d.
            tsum = tsum + dt
            nleft = nleft - 1
            iter_ = iter_ + 1
            dibp = dl[ibp - 1]
            dl[ibp - 1] = 0.0
            if dibp > 0.0:
                zibp = ul[ibp - 1] - xl[ibp - 1]
                xcpl[ibp - 1] = ul[ibp - 1]
                where[ibp - 1] = 2
            else:
                zibp = ll[ibp - 1] - xl[ibp - 1]
                xcpl[ibp - 1] = ll[ibp - 1]
                where[ibp - 1] = 1
            if verbose and iprint >= 100:
                out(UNIT_OUT, list_directed("Variable  ", ibp, "  is fixed.") + "\n")
            if nleft == 0 and nbreak == n:
                # All n variables are fixed: return with xcp as GCP.
                dtm = dt
                all_fixed = True
                break

            # Update the derivative information.
            nseg = nseg + 1
            dibp2 = dibp * dibp

            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp
            f2 = f2 - theta * dibp2

            if col > 0:
                _daxpy_list(col2, dt, p, c)
                pointr = head
                for j in range(col):
                    wbp[j] = wyl[ibp - 1][pointr - 1]
                    wbp[col + j] = theta * wsl[ibp - 1][pointr - 1]
                    pointr = pointr % m + 1
                info = _bmv(m, syl, wtl, col, wbp, v, info)
                if info != 0:
                    store()
                    return nseg, info
                wmc = _ddot_list(col2, c, v)
                wmp = _ddot_list(col2, p, v)
                wmw = _ddot_list(col2, wbp, v)
                _daxpy_list(col2, -dibp, wbp, p)
                f1 = f1 + dibp * wmc
                f2 = f2 + 2.0 * dibp * wmp - dibp2 * wmw

            f2 = max(epsmch * f2_org, f2)
            if nleft > 0:
                dtm = _div(-f1, f2)
                continue
            if bnded:
                f1 = 0.0
                f2 = 0.0
                dtm = 0.0
            else:
                dtm = _div(-f1, f2)
            break

    if not all_fixed:
        if verbose and iprint >= 99:
            out(UNIT_OUT, "\n")
            out(UNIT_OUT, list_directed("GCP found in this segment") + "\n")
            out(
                UNIT_OUT,
                "Piece    %s --f1, f2 at start point  %s %s\n"
                % (_fi(nseg, 3), _fd(f1, 11, 4), _fd(f2, 11, 4)),
            )
            out(
                UNIT_OUT,
                "Distance to the stationary point =  %s\n" % _fd(dtm, 11, 4),
            )
        if dtm <= 0.0:
            dtm = 0.0
        tsum = tsum + dtm
        # Move the free variables and those whose breakpoints are not reached.
        _daxpy_list(n, tsum, dl, xcpl)

    # c = c + dtm*p = W'(x^c - x), used by cmprlb.
    if col > 0:
        _daxpy_list(col2, dtm, p, c)
    store()
    if verbose and iprint > 100:
        out(UNIT_OUT, _cauchy_block(xcpl))
    if verbose and iprint >= 99:
        out(UNIT_OUT, "\n---------------- exit CAUCHY----------------------\n\n")
    return nseg, info


def _cmprlb(
    n: int,
    m: int,
    x: np.ndarray,
    g: np.ndarray,
    ws: np.ndarray,
    wy: np.ndarray,
    sy: np.ndarray,
    wt: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    wa: np.ndarray,
    index: np.ndarray,
    theta: float,
    col: int,
    head: int,
    nfree: int,
    cnstnd: bool,
    info: int,
) -> int:
    """``cmprlb``: ``r = -Z'B(xcp - xk) - Z'g`` from ``wa(2m+1) = W'(xcp - x)``.

    Returns ``info`` (-8 when the triangular system of ``bmv`` is singular).
    """
    if not cnstnd and col > 0:
        r[:] = -g
        return info
    free = index[:nfree].astype(np.intp) - 1
    r[:nfree] = -theta * (z[free] - x[free]) - g[free]
    m2 = 2 * m
    p = wa[0:m2].tolist()
    v = wa[m2 : 2 * m2].tolist()
    info = _bmv(m, sy.tolist(), wt.tolist(), col, v, p, info)
    wa[0:m2] = p
    if info != 0:
        return -8
    pointr = head
    for j in range(col):
        a1 = p[j]
        a2 = theta * p[col + j]
        r[:nfree] = r[:nfree] + wy[free, pointr - 1] * a1 + ws[free, pointr - 1] * a2
        pointr = pointr % m + 1
    return info


def _errclb(
    n: int,
    m: int,
    factr: float,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    task: str,
    info: int,
    k: int,
) -> tuple[str, int, int]:
    """``errclb``: check the input arguments.  Returns ``(task, info, k)``."""
    if n <= 0:
        task = "ERROR: N .LE. 0"
    if m <= 0:
        task = "ERROR: M .LE. 0"
    if factr < 0.0:
        task = "ERROR: FACTR .LT. 0"
    for i in range(n):
        if nbd[i] < 0 or nbd[i] > 3:
            task = "ERROR: INVALID NBD"
            info = -6
            k = i + 1
        if nbd[i] == 2:
            if l[i] > u[i]:
                task = "ERROR: NO FEASIBLE SOLUTION"
                info = -7
                k = i + 1
    return task, info, k


def _formk(
    n: int,
    nsub: int,
    ind: np.ndarray,
    nenter: int,
    ileave: int,
    indx2: np.ndarray,
    iupdat: int,
    updatd: bool,
    wn: np.ndarray,
    wn1: np.ndarray,
    m: int,
    ws: np.ndarray,
    wy: np.ndarray,
    sy: np.ndarray,
    theta: float,
    col: int,
    head: int,
) -> int:
    """``formk``: LEL' factorization of the indefinite subspace matrix

    ``K = [-D - Y'ZZ'Y/theta, L_a' - R_z'; L_a - R_z, theta*S'AA'S]``.

    ``wn1`` (``snd`` of ``mainlb``) keeps the inner products between calls;
    the factor is left in the upper triangle of ``wn``.  Returns ``info``:
    0, -1 or -2 when the first or the second Cholesky factorization fails.
    """
    # Lower triangular part of WN1 = [Y'ZZ'Y, L_a'+R_z'; L_a+R_z, S'AA'S].
    if updatd:
        if iupdat > m:
            # Shift the old part of WN1.
            for jy in range(1, m):
                js = m + jy
                wn1[jy - 1 : m - 1, jy - 1] = wn1[jy:m, jy]
                wn1[js - 1 : 2 * m - 1, js - 1] = wn1[js : 2 * m, js]
                wn1[m : 2 * m - 1, jy - 1] = wn1[m + 1 : 2 * m, jy]

        # New rows in blocks (1,1), (2,1) and (2,2).
        free = ind[:nsub].astype(np.intp) - 1
        fixed = ind[nsub:n].astype(np.intp) - 1
        iy = col
        is_ = m + col
        ipntr = head + col - 1
        if ipntr > m:
            ipntr = ipntr - m
        jpntr = head
        wy_free = wy[free, ipntr - 1]
        ws_fixed = ws[fixed, ipntr - 1]
        for jy in range(1, col + 1):
            js = m + jy
            temp1 = _ddot(wy_free, wy[free, jpntr - 1])
            temp2 = _ddot(ws_fixed, ws[fixed, jpntr - 1])
            temp3 = _ddot(ws_fixed, wy[fixed, jpntr - 1])
            wn1[iy - 1, jy - 1] = temp1
            wn1[is_ - 1, js - 1] = temp2
            wn1[is_ - 1, jy - 1] = temp3
            jpntr = jpntr % m + 1

        # New column in block (2,1).
        jy = col
        jpntr = head + col - 1
        if jpntr > m:
            jpntr = jpntr - m
        ipntr = head
        wy_free = wy[free, jpntr - 1]
        for i in range(1, col + 1):
            is_ = m + i
            temp3 = _ddot(ws[free, ipntr - 1], wy_free)
            ipntr = ipntr % m + 1
            wn1[is_ - 1, jy - 1] = temp3
        upcl = col - 1
    else:
        upcl = col

    # Modify the old parts in blocks (1,1) and (2,2) due to changes in the
    # set of free variables.
    enter = indx2[:nenter].astype(np.intp) - 1
    leave = indx2[ileave - 1 : n].astype(np.intp) - 1
    ipntr = head
    for iy in range(1, upcl + 1):
        is_ = m + iy
        jpntr = head
        for jy in range(1, iy + 1):
            js = m + jy
            temp1 = _ddot(wy[enter, ipntr - 1], wy[enter, jpntr - 1])
            temp2 = _ddot(ws[enter, ipntr - 1], ws[enter, jpntr - 1])
            temp3 = _ddot(wy[leave, ipntr - 1], wy[leave, jpntr - 1])
            temp4 = _ddot(ws[leave, ipntr - 1], ws[leave, jpntr - 1])
            wn1[iy - 1, jy - 1] = float(wn1[iy - 1, jy - 1]) + temp1 - temp3
            wn1[is_ - 1, js - 1] = float(wn1[is_ - 1, js - 1]) - temp2 + temp4
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1

    # Modify the old parts in block (2,1).
    ipntr = head
    for is_ in range(m + 1, m + upcl + 1):
        jpntr = head
        for jy in range(1, upcl + 1):
            temp1 = _ddot(ws[enter, ipntr - 1], wy[enter, jpntr - 1])
            temp3 = _ddot(ws[leave, ipntr - 1], wy[leave, jpntr - 1])
            if is_ <= jy + m:
                wn1[is_ - 1, jy - 1] = float(wn1[is_ - 1, jy - 1]) + temp1 - temp3
            else:
                wn1[is_ - 1, jy - 1] = float(wn1[is_ - 1, jy - 1]) - temp1 + temp3
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1

    # Upper triangle of WN = [D + Y'ZZ'Y/theta, -L_a' + R_z';
    #                         -L_a + R_z,       S'AA'S*theta].
    w = wn.tolist()
    w1 = wn1.tolist()
    syl = sy.tolist()
    try:
        for iy in range(1, col + 1):
            is_ = col + iy
            is1 = m + iy
            for jy in range(1, iy + 1):
                js = col + jy
                js1 = m + jy
                w[jy - 1][iy - 1] = _div(w1[iy - 1][jy - 1], theta)
                w[js - 1][is_ - 1] = w1[is1 - 1][js1 - 1] * theta
            for jy in range(1, iy):
                w[jy - 1][is_ - 1] = -w1[is1 - 1][jy - 1]
            for jy in range(iy, col + 1):
                w[jy - 1][is_ - 1] = w1[is1 - 1][jy - 1]
            w[iy - 1][iy - 1] = w[iy - 1][iy - 1] + syl[iy - 1][iy - 1]

        # Cholesky factor of the (1,1) block: LL' with L' in the upper triangle.
        info = _dpofa(w, col)
        if info != 0:
            return -1
        # L^-1(-L_a' + R_z') in the (1,2) block.
        col2 = 2 * col
        for js in range(col, col2):
            column = [w[i][js] for i in range(col)]
            info = _dtrsl(w, col, column, 0, 11)
            for i in range(col):
                w[i][js] = column[i]

        # S'AA'S*theta + (L^-1(-L_a'+R_z'))'(L^-1(-L_a'+R_z')) in the upper
        # triangle of the (2,2) block.
        for is_ in range(col, col2):
            for js in range(is_, col2):
                dot = 0.0
                for i in range(col):
                    dot = dot + w[i][is_] * w[i][js]
                w[is_][js] = w[is_][js] + dot

        # Cholesky factorization of the (2,2) block.
        info = _dpofa(w, col, col)
        if info != 0:
            return -2
        return 0
    finally:
        wn[:, :] = w


def _formt(
    m: int,
    wt: np.ndarray,
    sy: np.ndarray,
    ss: np.ndarray,
    col: int,
    theta: float,
) -> int:
    """``formt``: ``T = theta*S'S + L D^-1 L'`` and its Cholesky factor ``J'``
    in the upper triangle of ``wt``.  Returns ``info`` (0 or -3).
    """
    w = wt.tolist()
    syl = sy.tolist()
    ssl = ss.tolist()
    for j in range(col):
        w[0][j] = theta * ssl[0][j]
    for i in range(1, col):
        for j in range(i, col):
            k1 = min(i, j)
            ddum = 0.0
            for k in range(k1):
                ddum = ddum + _div(syl[i][k] * syl[j][k], syl[k][k])
            w[i][j] = ddum + theta * ssl[i][j]
    info = _dpofa(w, col)
    wt[:, :] = w
    if info != 0:
        info = -3
    return info


def _freev(
    n: int,
    nfree: int,
    index: np.ndarray,
    indx2: np.ndarray,
    iwhere: np.ndarray,
    updatd: bool,
    cnstnd: bool,
    iprint: int,
    iter_: int,
    out: OutputSink | None,
) -> tuple[int, int, int, bool]:
    """``freev``: entering and leaving variables, free and active index sets.

    Returns ``(nfree, nenter, ileave, wrk)``.
    """
    verbose = out is not None
    nenter = 0
    ileave = n + 1
    where = iwhere.tolist()
    idx = index.tolist()
    if iter_ > 0 and cnstnd:
        idx2 = indx2.tolist()
        for i in range(nfree):
            k = idx[i]
            if where[k - 1] > 0:
                ileave = ileave - 1
                idx2[ileave - 1] = k
                if verbose and iprint >= 100:
                    out(
                        UNIT_OUT,
                        list_directed(
                            "Variable ", k, " leaves the set of free variables"
                        )
                        + "\n",
                    )
        for i in range(nfree, n):
            k = idx[i]
            if where[k - 1] <= 0:
                nenter = nenter + 1
                idx2[nenter - 1] = k
                if verbose and iprint >= 100:
                    out(
                        UNIT_OUT,
                        list_directed(
                            "Variable ", k, " enters the set of free variables"
                        )
                        + "\n",
                    )
        indx2[:] = idx2
        if verbose and iprint >= 99:
            out(
                UNIT_OUT,
                list_directed(
                    n + 1 - ileave, " variables leave; ", nenter, " variables enter"
                )
                + "\n",
            )
    wrk = (ileave < n + 1) or (nenter > 0) or updatd

    # Index set of free and active variables at the GCP.
    nfree = 0
    iact = n + 1
    for i in range(n):
        if where[i] <= 0:
            nfree = nfree + 1
            idx[nfree - 1] = i + 1
        else:
            iact = iact - 1
            idx[iact - 1] = i + 1
    index[:] = idx
    if verbose and iprint >= 99:
        out(
            UNIT_OUT,
            list_directed(nfree, " variables are free at GCP ", iter_ + 1) + "\n",
        )
    return nfree, nenter, ileave, wrk


def _dcstep(
    stx: float,
    fx: float,
    dx: float,
    sty: float,
    fy: float,
    dy: float,
    stp: float,
    fp: float,
    dp: float,
    brackt: bool,
    stpmin: float,
    stpmax: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    """MINPACK-2 ``dcstep``: safeguarded step of the More-Thuente line search.

    Returns the updated ``(stx, fx, dx, sty, fy, dy, stp, brackt)``.
    """
    sgnd = dp * _div(dx, abs(dx))

    if fp > fx:
        # A higher function value: the minimum is bracketed.
        theta = _div(3.0 * (fx - fp), stp - stx) + dx + dp
        s = max(_fmax(abs(theta), abs(dx)), abs(dp))
        ratio = _div(theta, s)
        gamma = s * _sqrt(ratio * ratio - _div(dx, s) * _div(dp, s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = _div(p, q)
        stpc = stx + r * (stp - stx)
        stpq = stx + _div(_div(dx, _div(fx - fp, stp - stx) + dx), 2.0) * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # A lower function value and derivatives of opposite sign.
        theta = _div(3.0 * (fx - fp), stp - stx) + dx + dp
        s = max(_fmax(abs(theta), abs(dx)), abs(dp))
        ratio = _div(theta, s)
        gamma = s * _sqrt(ratio * ratio - _div(dx, s) * _div(dp, s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = _div(p, q)
        stpc = stp + r * (stx - stp)
        stpq = stp + _div(dp, dp - dx) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # A lower function value, derivatives of the same sign, and the
        # magnitude of the derivative decreases.
        theta = _div(3.0 * (fx - fp), stp - stx) + dx + dp
        s = max(_fmax(abs(theta), abs(dx)), abs(dp))
        ratio = _div(theta, s)
        gamma = s * _sqrt(max(0.0, ratio * ratio - _div(dx, s) * _div(dp, s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = _div(p, q)
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + _div(dp, dp - dx) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = _fmin(stp + _P66 * (sty - stp), stpf)
            else:
                stpf = _fmax(stp + _P66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = _fmin(stpmax, stpf)
            stpf = _fmax(stpmin, stpf)
    else:
        # A lower function value, derivatives of the same sign, and the
        # magnitude of the derivative does not decrease.
        if brackt:
            theta = _div(3.0 * (fp - fy), sty - stp) + dy + dp
            s = max(_fmax(abs(theta), abs(dy)), abs(dp))
            ratio = _div(theta, s)
            gamma = s * _sqrt(ratio * ratio - _div(dy, s) * _div(dp, s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = _div(p, q)
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    stp = stpf
    return stx, fx, dx, sty, fy, dy, stp, brackt


def _dcsrch(
    f: float,
    g: float,
    stp: float,
    ftol: float,
    gtol: float,
    xtol: float,
    stpmin: float,
    stpmax: float,
    task: str,
    isave: np.ndarray,
    dsave: np.ndarray,
) -> tuple[float, str]:
    """MINPACK-2 ``dcsrch``: one reverse-communication step of the line search.

    ``isave`` (2) and ``dsave`` (13) are the slices ``isave(43:44)`` and
    ``dsave(17:29)`` of the L-BFGS-B state.  Returns ``(stp, task)``.
    """
    if task[:5] == "START":
        if stp < stpmin:
            task = "ERROR: STP .LT. STPMIN"
        if stp > stpmax:
            task = "ERROR: STP .GT. STPMAX"
        if g >= 0.0:
            task = "ERROR: INITIAL G .GE. ZERO"
        if ftol < 0.0:
            task = "ERROR: FTOL .LT. ZERO"
        if gtol < 0.0:
            task = "ERROR: GTOL .LT. ZERO"
        if xtol < 0.0:
            task = "ERROR: XTOL .LT. ZERO"
        if stpmin < 0.0:
            task = "ERROR: STPMIN .LT. ZERO"
        if stpmax < stpmin:
            task = "ERROR: STPMAX .LT. STPMIN"

        # Exit if there are errors on input (the work arrays are not saved).
        if task[:5] == "ERROR":
            return stp, task

        brackt = False
        stage = 1
        finit = f
        ginit = g
        gtest = ftol * ginit
        width = stpmax - stpmin
        width1 = width / _P5

        stx = 0.0
        fx = finit
        gx = ginit
        sty = 0.0
        fy = finit
        gy = ginit
        stmin = 0.0
        stmax = stp + _XTRAPU * stp
        task = "FG"
    else:
        brackt = int(isave[0]) == 1
        stage = int(isave[1])
        ginit = float(dsave[0])
        gtest = float(dsave[1])
        gx = float(dsave[2])
        gy = float(dsave[3])
        finit = float(dsave[4])
        fx = float(dsave[5])
        fy = float(dsave[6])
        stx = float(dsave[7])
        sty = float(dsave[8])
        stmin = float(dsave[9])
        stmax = float(dsave[10])
        width = float(dsave[11])
        width1 = float(dsave[12])

        # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm
        # enters the second stage.
        ftest = finit + stp * gtest
        if stage == 1 and f <= ftest and g >= 0.0:
            stage = 2

        # Test for warnings.
        if brackt and (stp <= stmin or stp >= stmax):
            task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
        if brackt and stmax - stmin <= xtol * stmax:
            task = "WARNING: XTOL TEST SATISFIED"
        if stp == stpmax and f <= ftest and g <= gtest:
            task = "WARNING: STP = STPMAX"
        if stp == stpmin and (f > ftest or g >= gtest):
            task = "WARNING: STP = STPMIN"

        # Test for convergence.
        if f <= ftest and abs(g) <= gtol * (-ginit):
            task = "CONVERGENCE"

        if task[:4] != "WARN" and task[:4] != "CONV":
            # A modified function is used to predict the step during the
            # first stage if a lower function value has been obtained but the
            # decrease is not sufficient.
            if stage == 1 and f <= fx and f > ftest:
                fm = f - stp * gtest
                fxm = fx - stx * gtest
                fym = fy - sty * gtest
                gm = g - gtest
                gxm = gx - gtest
                gym = gy - gtest

                stx, fxm, gxm, sty, fym, gym, stp, brackt = _dcstep(
                    stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax
                )

                fx = fxm + stx * gtest
                fy = fym + sty * gtest
                gx = gxm + gtest
                gy = gym + gtest
            else:
                stx, fx, gx, sty, fy, gy, stp, brackt = _dcstep(
                    stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax
                )

            # Decide if a bisection step is needed.
            if brackt:
                if abs(sty - stx) >= _P66 * width1:
                    stp = stx + _P5 * (sty - stx)
                width1 = width
                width = abs(sty - stx)

            # Set the minimum and maximum steps allowed for stp.
            if brackt:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stp + _XTRAPL * (stp - stx)
                stmax = stp + _XTRAPU * (stp - stx)

            # Force the step to be within the bounds stpmax and stpmin.
            stp = _fmax(stp, stpmin)
            stp = _fmin(stp, stpmax)

            # If further progress is not possible, let stp be the best point
            # obtained during the search.
            if (brackt and (stp <= stmin or stp >= stmax)) or (
                brackt and stmax - stmin <= xtol * stmax
            ):
                stp = stx

            # Obtain another function and derivative.
            task = "FG"

    isave[0] = 1 if brackt else 0
    isave[1] = stage
    dsave[0] = ginit
    dsave[1] = gtest
    dsave[2] = gx
    dsave[3] = gy
    dsave[4] = finit
    dsave[5] = fx
    dsave[6] = fy
    dsave[7] = stx
    dsave[8] = sty
    dsave[9] = stmin
    dsave[10] = stmax
    dsave[11] = width
    dsave[12] = width1
    return stp, task


def _lnsrlb(
    n: int,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    x: np.ndarray,
    f: float,
    fold: float,
    gd: float,
    gdold: float,
    g: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    stp: float,
    dnorm: float,
    dtd: float,
    xstep: float,
    stpmx: float,
    iter_: int,
    ifun: int,
    iback: int,
    nfgv: int,
    info: int,
    task: str,
    boxed: bool,
    cnstnd: bool,
    csave: str,
    isave: np.ndarray,
    dsave: np.ndarray,
    out: OutputSink | None,
) -> tuple[
    float, float, float, float, float, float, float, float, int, int, int, int, str, str
]:
    """``lnsrlb``: line search by ``dcsrch`` with all trial points feasible.

    On a new line search ``t`` and ``r`` receive the current ``x`` and ``g``;
    when a trial point is requested ``x`` is overwritten with it.  Returns
    ``(fold, gd, gdold, stp, dnorm, dtd, xstep, stpmx, ifun, iback, nfgv,
    info, task, csave)``; ``info = -4`` flags an ascent direction.
    """
    if task[:5] != "FG_LN":
        dtd = _ddot(d, d)
        dnorm = _sqrt(dtd)

        # Maximum step length.
        stpmx = _BIG
        if cnstnd:
            if iter_ == 0:
                stpmx = 1.0
            else:
                dl = d.tolist()
                xl = x.tolist()
                ll = l.tolist()
                ul = u.tolist()
                for i in range(n):
                    a1 = dl[i]
                    if nbd[i] != 0:
                        if a1 < 0.0 and nbd[i] <= 2:
                            a2 = ll[i] - xl[i]
                            if a2 >= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx < a2:
                                stpmx = a2 / a1
                        elif a1 > 0.0 and nbd[i] >= 2:
                            a2 = ul[i] - xl[i]
                            if a2 <= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx > a2:
                                stpmx = a2 / a1

        if iter_ == 0 and not boxed:
            stp = _fmin(_div(1.0, dnorm), stpmx)
        else:
            stp = 1.0

        t[:] = x
        r[:] = g
        fold = f
        ifun = 0
        iback = 0
        csave = "START"

    gd = _ddot(g, d)
    if ifun == 0:
        gdold = gd
        if gd >= 0.0:
            # The directional derivative is >= 0: line search is impossible.
            # (This message is not controlled by iprint.)
            if out is not None:
                out(
                    UNIT_OUT,
                    list_directed(" ascent direction in projection gd = ", gd) + "\n",
                )
            info = -4
            return (
                fold, gd, gdold, stp, dnorm, dtd, xstep, stpmx, ifun, iback, nfgv,
                info, task, csave,
            )

    stp, csave = _dcsrch(
        f, gd, stp, _FTOL, _GTOL, _XTOL, 0.0, stpmx, csave, isave, dsave
    )

    xstep = stp * dnorm
    if csave[:4] != "CONV" and csave[:4] != "WARN":
        task = "FG_LNSRCH"
        ifun = ifun + 1
        nfgv = nfgv + 1
        iback = ifun - 1
        if stp == 1.0:
            x[:] = z
        else:
            x[:] = stp * d + t
    else:
        task = "NEW_X"
    return (
        fold, gd, gdold, stp, dnorm, dtd, xstep, stpmx, ifun, iback, nfgv, info,
        task, csave,
    )


def _matupd(
    n: int,
    m: int,
    ws: np.ndarray,
    wy: np.ndarray,
    sy: np.ndarray,
    ss: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    itail: int,
    iupdat: int,
    col: int,
    head: int,
    rr: float,
    dr: float,
    stp: float,
    dtd: float,
) -> tuple[int, int, int, float]:
    """``matupd``: update ``WS``, ``WY`` and form the middle matrix of ``B``.

    Returns ``(itail, col, head, theta)``.
    """
    if iupdat <= m:
        col = iupdat
        itail = (head + iupdat - 2) % m + 1
    else:
        itail = itail % m + 1
        head = head % m + 1

    ws[:, itail - 1] = d
    wy[:, itail - 1] = r

    theta = _div(rr, dr)

    # Upper triangle of SS and lower triangle of SY.
    if iupdat > m:
        # Move the old information.
        for j in range(1, col):
            ss[0:j, j - 1] = ss[1 : j + 1, j]
            sy[j - 1 : col - 1, j - 1] = sy[j:col, j]
    # The last row of SY and the last column of SS.
    pointr = head
    for j in range(1, col):
        sy[col - 1, j - 1] = _ddot(d, wy[:, pointr - 1])
        ss[j - 1, col - 1] = _ddot(ws[:, pointr - 1], d)
        pointr = pointr % m + 1
    if stp == 1.0:
        ss[col - 1, col - 1] = dtd
    else:
        ss[col - 1, col - 1] = stp * stp * dtd
    sy[col - 1, col - 1] = dr
    return itail, col, head, theta


def _projgr(
    n: int,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    x: np.ndarray,
    g: np.ndarray,
) -> float:
    """``projgr``: infinity norm of the projected gradient."""
    sbgnrm = 0.0
    xl = x.tolist()
    gl = g.tolist()
    ll = l.tolist()
    ul = u.tolist()
    for i in range(n):
        gi = gl[i]
        if nbd[i] != 0:
            if gi < 0.0:
                if nbd[i] >= 2:
                    gi = _fmax(xl[i] - ul[i], gi)
            else:
                if nbd[i] <= 2:
                    gi = _fmin(xl[i] - ll[i], gi)
        sbgnrm = max(sbgnrm, abs(gi))
    return sbgnrm


def _subsm(
    n: int,
    m: int,
    nsub: int,
    ind: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    x: np.ndarray,
    d: np.ndarray,
    xp: np.ndarray,
    ws: np.ndarray,
    wy: np.ndarray,
    theta: float,
    xx: np.ndarray,
    gg: np.ndarray,
    col: int,
    head: int,
    iword: int,
    wv: np.ndarray,
    wn: np.ndarray,
    iprint: int,
    info: int,
    out: OutputSink | None,
) -> tuple[int, int]:
    """``subsm``: subspace minimization along the Newton direction with the
    projection and backtracking safeguard of L-BFGS-B 3.0.

    On entry ``x`` is the Cauchy point and ``d`` the reduced gradient
    (``z`` and ``r`` of ``mainlb``); on exit ``x`` is the subspace minimizer
    and ``d`` the Newton direction.  ``xx`` and ``gg`` are the current iterate
    and gradient.  ``wv`` is the first quarter (2m) of the 8m work array.
    Returns ``(iword, info)``.
    """
    if nsub <= 0:
        return iword, info
    verbose = out is not None
    if verbose and iprint >= 99:
        out(UNIT_OUT, "\n----------------SUBSM entered-----------------\n\n")

    free = ind[:nsub].astype(np.intp) - 1
    m2 = 2 * m
    col2 = 2 * col

    # wv = W'Zd.
    wvl = wv[0:m2].tolist()
    pointr = head
    dsub = d[:nsub]
    for i in range(col):
        temp1 = _ddot(wy[free, pointr - 1], dsub)
        temp2 = _ddot(ws[free, pointr - 1], dsub)
        wvl[i] = temp1
        wvl[col + i] = theta * temp2
        pointr = pointr % m + 1

    # wv := K^(-1) wv.
    wnl = wn.tolist()
    info = _dtrsl(wnl, col2, wvl, 0, 11)
    if info != 0:
        wv[0:m2] = wvl
        return iword, info
    for i in range(col):
        wvl[i] = -wvl[i]
    info = _dtrsl(wnl, col2, wvl, 0, 1)
    wv[0:m2] = wvl
    if info != 0:
        return iword, info

    # d = (1/theta) d + (1/theta**2) Z'W wv.
    pointr = head
    for jy in range(col):
        js = col + jy
        dsub[:] = (
            dsub
            + wy[free, pointr - 1] * wvl[jy] / theta
            + ws[free, pointr - 1] * wvl[js]
        )
        pointr = pointr % m + 1
    dsub[:] = _div(1.0, theta) * dsub

    # Try the projection; d is the Newton direction.
    iword = 0
    xp[:] = x
    xl = x.tolist()
    dl = dsub.tolist()
    ll = l.tolist()
    ul = u.tolist()
    idx = free.tolist()
    for i in range(nsub):
        k = idx[i]
        dk = dl[i]
        xk = xl[k]
        nb = nbd[k]
        if nb != 0:
            if nb == 1:
                xl[k] = max(ll[k], xk + dk)
                if xl[k] == ll[k]:
                    iword = 1
            elif nb == 2:
                xk = max(ll[k], xk + dk)
                xl[k] = _fmin(ul[k], xk)
                if xl[k] == ll[k] or xl[k] == ul[k]:
                    iword = 1
            elif nb == 3:
                xl[k] = min(ul[k], xk + dk)
                if xl[k] == ul[k]:
                    iword = 1
        else:
            xl[k] = xk + dk
    x[:] = xl

    if iword != 0:
        # Check the sign of the directional derivative.
        dd_p = _ddot(x - xx, gg)
        if dd_p > 0.0:
            x[:] = xp
            if verbose:
                out(
                    UNIT_OUT,
                    list_directed(" Positive dir derivative in projection ") + "\n",
                )
                out(UNIT_OUT, list_directed(" Using the backtracking step ") + "\n")
            xl = x.tolist()
            alpha = 1.0
            temp1 = alpha
            ibd = 0
            for i in range(nsub):
                k = idx[i]
                dk = dl[i]
                nb = nbd[k]
                if nb != 0:
                    if dk < 0.0 and nb <= 2:
                        temp2 = ll[k] - xl[k]
                        if temp2 >= 0.0:
                            temp1 = 0.0
                        elif dk * alpha < temp2:
                            temp1 = temp2 / dk
                    elif dk > 0.0 and nb >= 2:
                        temp2 = ul[k] - xl[k]
                        if temp2 <= 0.0:
                            temp1 = 0.0
                        elif dk * alpha > temp2:
                            temp1 = temp2 / dk
                    if temp1 < alpha:
                        alpha = temp1
                        ibd = i + 1

            if alpha < 1.0:
                dk = dl[ibd - 1]
                k = idx[ibd - 1]
                if dk > 0.0:
                    xl[k] = ul[k]
                    dl[ibd - 1] = 0.0
                elif dk < 0.0:
                    xl[k] = ll[k]
                    dl[ibd - 1] = 0.0
            for i in range(nsub):
                k = idx[i]
                xl[k] = xl[k] + alpha * dl[i]
            x[:] = xl
            dsub[:] = dl

    if verbose and iprint >= 99:
        out(UNIT_OUT, "\n----------------exit SUBSM --------------------\n\n")
    return iword, info


# ---------------------------------------------------------------------------
# mainlb and setulb
# ---------------------------------------------------------------------------

# Statement labels of mainlb.
_L_FG_START = 0
_L111 = 111
_L222 = 222
_L333 = 333
_L555 = 555
_L666 = 666
_L777 = 777
_L999 = 999
_L1000 = 1000


def _mainlb(
    n: int,
    m: int,
    x: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    nbd: list,
    f: float,
    g: np.ndarray,
    factr: float,
    pgtol: float,
    ws: np.ndarray,
    wy: np.ndarray,
    sy: np.ndarray,
    ss: np.ndarray,
    wt: np.ndarray,
    wn: np.ndarray,
    snd: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    xp: np.ndarray,
    wa: np.ndarray,
    index: np.ndarray,
    iwhere: np.ndarray,
    indx2: np.ndarray,
    state: LBFGSBState,
    iprint: int,
    out: OutputSink | None,
    clock: Clock | None,
) -> float:
    """``mainlb``: the L-BFGS-B iteration in reverse communication.

    ``isave(1:23)`` of the Fortran routine is ``state.isave[21:44]``.
    Returns ``f`` (see ``setulb``).
    """
    task = state.task
    csave = state.csave
    lsave = state.lsave
    isave = state.isave[21:44]
    dsave = state.dsave
    verbose = out is not None

    def timer() -> float:
        # timer.f returns cpu_time through a single-precision variable.
        if clock is None:
            return 0.0
        return float(np.float32(clock()))

    # Locals that the Fortran does not keep between calls.
    wrk = False
    xstep = 0.0
    k = 0

    if task.rstrip(" ") == "START":
        epsmch = _EPSMCH
        time1 = timer()

        # Counters and scalars of the limited memory BFGS matrices.
        col = 0
        head = 1
        theta = 1.0
        iupdat = 0
        updatd = False
        iback = 0
        itail = 0
        iword = 0
        nact = 0
        ileave = 0
        nenter = 0
        fold = 0.0
        dnorm = 0.0
        cpu1 = 0.0
        gd = 0.0
        stpmx = 0.0
        sbgnrm = 0.0
        stp = 0.0
        gdold = 0.0
        dtd = 0.0

        # Operation counts.
        iter_ = 0
        nfgv = 0
        nseg = 0
        nintol = 0
        nskip = 0
        nfree = n
        ifun = 0
        # Stopping tolerance.
        tol = factr * epsmch

        # Running time.
        cachyt = 0.0
        sbtime = 0.0
        lnscht = 0.0

        # 'info' records the termination information.
        info = 0

        # The Fortran opens 'iterate.dat' on this unit when iprint >= 1.
        itfile = UNIT_ITERATE

        # Check the input arguments for errors.
        task, info, k = _errclb(n, m, factr, l, u, nbd, task, info, k)
        if task[:5] == "ERROR":
            _prn3lb(
                n, x, f, task, iprint, info, itfile, iter_, nfgv, nintol, nskip,
                nact, sbgnrm, 0.0, nseg, "---", iback, stp, xstep, k, cachyt,
                sbtime, lnscht, out,
            )
            # The Fortran returns here without saving its local variables.
            state.task = task
            return f

        _prn1lb(n, m, l, u, x, iprint, itfile, epsmch, out)

        # Initialize iwhere and project x onto the feasible set.
        prjctd, cnstnd, boxed = _active(n, l, u, nbd, x, iwhere, iprint, out)
        label = _L_FG_START
    else:
        # Restore local variables.
        prjctd = bool(lsave[0])
        cnstnd = bool(lsave[1])
        boxed = bool(lsave[2])
        updatd = bool(lsave[3])

        nintol = int(isave[0])
        itfile = int(isave[2])
        iback = int(isave[3])
        nskip = int(isave[4])
        head = int(isave[5])
        col = int(isave[6])
        itail = int(isave[7])
        iter_ = int(isave[8])
        iupdat = int(isave[9])
        nseg = int(isave[11])
        nfgv = int(isave[12])
        info = int(isave[13])
        ifun = int(isave[14])
        iword = int(isave[15])
        nfree = int(isave[16])
        nact = int(isave[17])
        ileave = int(isave[18])
        nenter = int(isave[19])

        theta = float(dsave[0])
        fold = float(dsave[1])
        tol = float(dsave[2])
        dnorm = float(dsave[3])
        epsmch = float(dsave[4])
        cpu1 = float(dsave[5])
        cachyt = float(dsave[6])
        sbtime = float(dsave[7])
        lnscht = float(dsave[8])
        time1 = float(dsave[9])
        gd = float(dsave[10])
        stpmx = float(dsave[11])
        sbgnrm = float(dsave[12])
        stp = float(dsave[13])
        gdold = float(dsave[14])
        dtd = float(dsave[15])

        # Go to the point where execution is to resume.
        if task[:5] == "FG_LN":
            label = _L666
        elif task[:5] == "NEW_X":
            label = _L777
        elif task[:5] == "FG_ST":
            label = _L111
        elif task[:4] == "STOP":
            if task[6:9] == "CPU":
                # Restore the previous iterate.
                x[:] = t
                g[:] = r
                f = fold
            label = _L999
        else:
            label = _L_FG_START

    while True:
        if label == _L_FG_START:
            # Compute f0 and g0: return to the driver, re-enter at 111.
            task = "FG_START"
            label = _L1000

        elif label == _L111:
            nfgv = 1

            # Infinity norm of the (-) projected gradient.
            sbgnrm = _projgr(n, l, u, nbd, x, g)

            if verbose and iprint >= 1:
                out(UNIT_OUT, _iterate_line(iter_, f, sbgnrm))
                out(
                    itfile,
                    " %s %s     -     -   -     -     -        -    %s %s\n"
                    % (_fi(iter_, 4), _fi(nfgv, 4), _fd(sbgnrm, 10, 3), _fd(f, 10, 3)),
                )
            if sbgnrm <= pgtol:
                task = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
                label = _L999
            else:
                label = _L222

        elif label == _L222:
            # ----------------- the beginning of the loop -----------------
            if verbose and iprint >= 99:
                out(UNIT_OUT, "\n\nITERATION %s\n" % _fi(iter_ + 1, 5))
            iword = -1

            if not cnstnd and col > 0:
                # Skip the search for the GCP.
                z[:] = x
                wrk = updatd
                nseg = 0
                label = _L333
                continue

            # Generalized Cauchy point.
            cpu1 = timer()
            nseg, info = _cauchy(
                n, x, l, u, nbd, g, indx2, iwhere, t, d, z, m, wy, ws, sy, wt,
                theta, col, head, wa, nseg, iprint, sbgnrm, info, epsmch, out,
            )
            if info != 0:
                # Singular triangular system: refresh the lbfgs memory.
                if verbose and iprint >= 1:
                    out(UNIT_OUT, _REFRESH_SINGULAR)
                info = 0
                col = 0
                head = 1
                theta = 1.0
                iupdat = 0
                updatd = False
                cpu2 = timer()
                cachyt = cachyt + cpu2 - cpu1
                label = _L222
                continue
            cpu2 = timer()
            cachyt = cachyt + cpu2 - cpu1
            nintol = nintol + nseg

            # Entering and leaving variables; free and active sets at the GCP.
            nfree, nenter, ileave, wrk = _freev(
                n, nfree, index, indx2, iwhere, updatd, cnstnd, iprint, iter_, out
            )
            nact = n - nfree
            label = _L333

        elif label == _L333:
            # No free variables or B = theta*I: skip the subspace minimization.
            if nfree == 0 or col == 0:
                label = _L555
                continue

            cpu1 = timer()

            # LEL^T factorization of the indefinite matrix K.
            if wrk:
                info = _formk(
                    n, nfree, index, nenter, ileave, indx2, iupdat, updatd, wn,
                    snd, m, ws, wy, sy, theta, col, head,
                )
            if info != 0:
                # Nonpositive definiteness in the Cholesky factorization:
                # refresh the lbfgs memory and restart the iteration.
                if verbose and iprint >= 1:
                    out(UNIT_OUT, _REFRESH_FORMK)
                info = 0
                col = 0
                head = 1
                theta = 1.0
                iupdat = 0
                updatd = False
                cpu2 = timer()
                sbtime = sbtime + cpu2 - cpu1
                label = _L222
                continue

            # r = -Z'B(xcp-xk) - Z'g (using wa(2m+1) = W'(xcp-x) from cauchy).
            info = _cmprlb(
                n, m, x, g, ws, wy, sy, wt, z, r, wa, index, theta, col, head,
                nfree, cnstnd, info,
            )
            if info == 0:
                # The direct method.
                iword, info = _subsm(
                    n, m, nfree, index, l, u, nbd, z, r, xp, ws, wy, theta, x, g,
                    col, head, iword, wa, wn, iprint, info, out,
                )
            if info != 0:
                # Singular triangular system: refresh the lbfgs memory and
                # restart the iteration.
                if verbose and iprint >= 1:
                    out(UNIT_OUT, _REFRESH_SINGULAR)
                info = 0
                col = 0
                head = 1
                theta = 1.0
                iupdat = 0
                updatd = False
                cpu2 = timer()
                sbtime = sbtime + cpu2 - cpu1
                label = _L222
                continue

            cpu2 = timer()
            sbtime = sbtime + cpu2 - cpu1
            label = _L555

        elif label == _L555:
            # Search direction d := z - x.
            d[:] = z - x
            cpu1 = timer()
            label = _L666

        elif label == _L666:
            (
                fold, gd, gdold, stp, dnorm, dtd, xstep, stpmx, ifun, iback,
                nfgv, info, task, csave,
            ) = _lnsrlb(
                n, l, u, nbd, x, f, fold, gd, gdold, g, d, r, t, z, stp, dnorm,
                dtd, xstep, stpmx, iter_, ifun, iback, nfgv, info, task, boxed,
                cnstnd, csave, isave[21:23], dsave[16:29], out,
            )
            if info != 0 or iback >= 20:
                # Restore the previous iterate.
                x[:] = t
                g[:] = r
                f = fold
                if col == 0:
                    # Abnormal termination.
                    if info == 0:
                        info = -9
                        # Restore the actual number of f and g evaluations.
                        nfgv = nfgv - 1
                        ifun = ifun - 1
                        iback = iback - 1
                    task = "ABNORMAL_TERMINATION_IN_LNSRCH"
                    iter_ = iter_ + 1
                    label = _L999
                else:
                    # Refresh the lbfgs memory and restart the iteration.
                    if verbose and iprint >= 1:
                        out(UNIT_OUT, _REFRESH_LNSRCH)
                    if info == 0:
                        nfgv = nfgv - 1
                    info = 0
                    col = 0
                    head = 1
                    theta = 1.0
                    iupdat = 0
                    updatd = False
                    task = "RESTART_FROM_LNSRCH"
                    cpu2 = timer()
                    lnscht = lnscht + cpu2 - cpu1
                    label = _L222
            elif task[:5] == "FG_LN":
                # Return to the driver for f and g; re-enter at 666.
                label = _L1000
            else:
                # Quantities related to the new x.
                cpu2 = timer()
                lnscht = lnscht + cpu2 - cpu1
                iter_ = iter_ + 1

                # Infinity norm of the projected (-)gradient.
                sbgnrm = _projgr(n, l, u, nbd, x, g)

                _prn2lb(
                    x, f, g, iprint, itfile, iter_, nfgv, nact, sbgnrm, nseg,
                    iword, iback, stp, xstep, out,
                )
                label = _L1000

        elif label == _L777:
            # Test for termination.
            if sbgnrm <= pgtol:
                task = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
                label = _L999
                continue

            ddum = max(_fmax(abs(fold), abs(f)), 1.0)
            if (fold - f) <= tol * ddum:
                task = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
                if iback >= 10:
                    # A warning for more than 10 evaluations in the line search.
                    info = -5
                label = _L999
                continue

            # d = newx - oldx, r = newg - oldg, rr = y'y and dr = y's.
            r[:] = g - r
            rr = _ddot(r, r)
            if stp == 1.0:
                dr = gd - gdold
                ddum = -gdold
            else:
                dr = (gd - gdold) * stp
                d[:] = stp * d
                ddum = -gdold * stp

            if dr <= epsmch * ddum:
                # Skip the L-BFGS update.
                nskip = nskip + 1
                updatd = False
                if verbose and iprint >= 1:
                    out(
                        UNIT_OUT,
                        "  ys=%s  -gs=%s BFGS update SKIPPED\n"
                        % (_fe(dr, 10, 3), _fe(ddum, 10, 3)),
                    )
                label = _L222
                continue

            # Update the L-BFGS matrix.
            updatd = True
            iupdat = iupdat + 1

            # Matrices WS and WY and the middle matrix in B.
            itail, col, head, theta = _matupd(
                n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head, rr, dr, stp,
                dtd,
            )

            # Upper half of T = theta*SS + L*D^(-1)*L' and its Cholesky factor.
            info = _formt(m, wt, sy, ss, col, theta)

            if info != 0:
                # Nonpositive definiteness in the Cholesky factorization:
                # refresh the lbfgs memory and restart the iteration.
                if verbose and iprint >= 1:
                    out(UNIT_OUT, _REFRESH_FORMT)
                info = 0
                col = 0
                head = 1
                theta = 1.0
                iupdat = 0
                updatd = False
            # -------------------- the end of the loop --------------------
            label = _L222

        elif label == _L999:
            time2 = timer()
            time = time2 - time1
            # 'word' is an unsaved local of the Fortran routine and undefined
            # here; the status of the current iteration is printed instead.
            _prn3lb(
                n, x, f, task, iprint, info, itfile, iter_, nfgv, nintol, nskip,
                nact, sbgnrm, time, nseg, _subspace_word(iword), iback, stp,
                xstep, k, cachyt, sbtime, lnscht, out,
            )
            label = _L1000

        else:
            break

    # Save local variables.
    lsave[0] = prjctd
    lsave[1] = cnstnd
    lsave[2] = boxed
    lsave[3] = updatd

    isave[0] = nintol
    isave[2] = itfile
    isave[3] = iback
    isave[4] = nskip
    isave[5] = head
    isave[6] = col
    isave[7] = itail
    isave[8] = iter_
    isave[9] = iupdat
    isave[11] = nseg
    isave[12] = nfgv
    isave[13] = info
    isave[14] = ifun
    isave[15] = iword
    isave[16] = nfree
    isave[17] = nact
    isave[18] = ileave
    isave[19] = nenter

    dsave[0] = theta
    dsave[1] = fold
    dsave[2] = tol
    dsave[3] = dnorm
    dsave[4] = epsmch
    dsave[5] = cpu1
    dsave[6] = cachyt
    dsave[7] = sbtime
    dsave[8] = lnscht
    dsave[9] = time1
    dsave[10] = gd
    dsave[11] = stpmx
    dsave[12] = sbgnrm
    dsave[13] = stp
    dsave[14] = gdold
    dsave[15] = dtd

    state.task = task
    state.csave = csave
    return f


def setulb(
    state: LBFGSBState,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nbd: np.ndarray,
    f: float,
    g: np.ndarray,
    factr: float,
    pgtol: float,
    *,
    iprint: int = -1,
    output: OutputSink | None = None,
    clock: Clock | None = None,
) -> float:
    """One reverse-communication call of L-BFGS-B (``setulb`` of ``lbfgsb.f``).

    Parameters
    ----------
    state:
        Persistent work arrays; ``state.task`` is the job on entry
        (``'START'`` for the first call) and the request on exit:
        ``'FG_START'`` / ``'FG_LNSRCH'`` (supply ``f`` and ``g`` at ``x`` and
        call again), ``'NEW_X'`` (an iteration is complete, call again),
        ``'CONVERGENCE: ...'``, ``'ABNORMAL_TERMINATION_IN_LNSRCH'`` or
        ``'ERROR: ...'``.  A caller may set ``'STOP...'`` to end the run.
    x:
        ``float64`` array of shape ``(n,)``, modified in place.  The unit is
        the caller's; ``domove.f90`` passes the coordinates of the movable
        atoms as ``(x1, y1, z1, x2, ...)`` in bohr.
    lower, upper:
        Bounds, shape ``(n,)``; only the entries selected by ``nbd`` are read.
    nbd:
        Integer array of shape ``(n,)``: 0 unbounded, 1 lower bound only,
        2 both bounds, 3 upper bound only.  ``domove.f90`` uses 2 everywhere
        with ``x0 -/+ stepmax`` when ``stepmax > 0`` and 0 otherwise.
    f:
        Function value at ``x`` (total energy in Ry in PARSEC); read when the
        task on entry starts with ``FG``.
    g:
        Gradient at ``x`` (minus the forces, Ry/bohr), ``float64`` of shape
        ``(n,)``.  Overwritten with the gradient of the previous iterate when
        the routine restores that iterate.
    factr, pgtol:
        Stopping tolerances: ``(f_k - f_k+1)/max(|f_k|, |f_k+1|, 1) <=
        factr*epsmch`` and ``max|proj g_i| <= pgtol``.  PARSEC passes
        ``1e-15`` for both.
    iprint:
        Print level of the Fortran routine; only used when ``output`` is
        given.  ``domove.f90`` passes 1, or 100 for a PARSEC output level of
        3 or more.
    output:
        Optional ``output(unit, text)`` callback that receives the text the
        Fortran writes: unit ``UNIT_OUT`` (9) is the report (``out.*`` in
        PARSEC) and unit ``UNIT_ITERATE`` (8) is ``iterate.dat``, which the
        Fortran opens anew on the ``'START'`` call when ``iprint >= 1``.
        Each call carries the complete records of one WRITE statement.  The
        numerical path does not depend on ``iprint`` or ``output``.
    clock:
        Optional CPU-time function (for example ``time.process_time``) for
        the timing slots ``dsave(6:10)`` and the printed times.  Without it
        those slots stay zero.

    Returns
    -------
    float
        ``f`` on exit: the input value, or the function value of the restored
        previous iterate after a failed line search or ``'STOP: CPU'``.
    """
    n = state.n
    m = state.m
    if not isinstance(x, np.ndarray) or x.dtype != np.float64 or x.shape != (n,):
        raise ValueError(f"x must be a float64 array of shape ({n},)")
    if not isinstance(g, np.ndarray) or g.dtype != np.float64 or g.shape != (n,):
        raise ValueError(f"g must be a float64 array of shape ({n},)")
    if not x.flags.writeable or not g.flags.writeable:
        raise ValueError("x and g are modified in place and must be writeable")
    l = np.asarray(lower, dtype=np.float64)
    u = np.asarray(upper, dtype=np.float64)
    nbd_array = np.asarray(nbd)
    if l.shape != (n,) or u.shape != (n,) or nbd_array.shape != (n,):
        raise ValueError(f"lower, upper and nbd must have shape ({n},)")
    if nbd_array.dtype.kind not in "iu":
        raise ValueError("nbd must be an integer array")
    nbd_list = nbd_array.tolist()
    f = float(f)
    factr = float(factr)
    pgtol = float(pgtol)
    iprint = int(iprint)

    isave = state.isave
    if state.task.rstrip(" ") == "START":
        isave[0] = m * n
        isave[1] = m**2
        isave[2] = 4 * m**2
        isave[3] = 1  # ws      m*n
        isave[4] = isave[3] + isave[0]  # wy      m*n
        isave[5] = isave[4] + isave[0]  # wsy     m**2
        isave[6] = isave[5] + isave[1]  # wss     m**2
        isave[7] = isave[6] + isave[1]  # wt      m**2
        isave[8] = isave[7] + isave[1]  # wn      4*m**2
        isave[9] = isave[8] + isave[2]  # wsnd    4*m**2
        isave[10] = isave[9] + isave[2]  # wz      n
        isave[11] = isave[10] + n  # wr      n
        isave[12] = isave[11] + n  # wd      n
        isave[13] = isave[12] + n  # wt      n
        isave[14] = isave[13] + n  # wxp     n
        isave[15] = isave[14] + n  # wa      8*m
    lws = int(isave[3]) - 1
    lwy = int(isave[4]) - 1
    lsy = int(isave[5]) - 1
    lss = int(isave[6]) - 1
    lwt = int(isave[7]) - 1
    lwn = int(isave[8]) - 1
    lsnd = int(isave[9]) - 1
    lz = int(isave[10]) - 1
    lr = int(isave[11]) - 1
    ld = int(isave[12]) - 1
    lt = int(isave[13]) - 1
    lxp = int(isave[14]) - 1
    lwa = int(isave[15]) - 1

    wa = state.wa
    blocks = (
        (lws, m * n),
        (lwy, m * n),
        (lsy, m * m),
        (lss, m * m),
        (lwt, m * m),
        (lwn, 4 * m * m),
        (lsnd, 4 * m * m),
        (lz, n),
        (lr, n),
        (ld, n),
        (lt, n),
        (lxp, n),
        (lwa, 8 * m),
    )
    for begin, length in blocks:
        if begin < 0 or begin + length > wa.shape[0]:
            raise ValueError(
                "state.isave(4:16) does not partition a work array of this size; "
                "the state does not belong to this n and m or was never started"
            )

    def matrix(begin: int, rows: int, columns: int) -> np.ndarray:
        return wa[begin : begin + rows * columns].reshape((rows, columns), order="F")

    iwa = state.iwa
    with np.errstate(all="ignore"):
        return _mainlb(
            n,
            m,
            x,
            l,
            u,
            nbd_list,
            f,
            g,
            factr,
            pgtol,
            matrix(lws, n, m),
            matrix(lwy, n, m),
            matrix(lsy, m, m),
            matrix(lss, m, m),
            matrix(lwt, m, m),
            matrix(lwn, 2 * m, 2 * m),
            matrix(lsnd, 2 * m, 2 * m),
            wa[lz : lz + n],
            wa[lr : lr + n],
            wa[ld : ld + n],
            wa[lt : lt + n],
            wa[lxp : lxp + n],
            wa[lwa : lwa + 8 * m],
            iwa[0:n],
            iwa[n : 2 * n],
            iwa[2 * n : 3 * n],
            state,
            iprint,
            output,
            clock,
        )


# ---------------------------------------------------------------------------
# relax_restart.dat
# ---------------------------------------------------------------------------


def format_restart_block(
    state: LBFGSBState,
    lower: np.ndarray,
    upper: np.ndarray,
    stepmax: float,
) -> str:
    """Return the L-BFGS part of ``relax_restart.dat`` as ``domove.f90`` writes it.

    The text starts with the record ``mmax, stepmax`` that follows the line
    `` LBFGS work variables follow`` and holds, one value per record: task and
    csave (60 columns each), ``lsave(1:4)``, ``lower(j), upper(j)`` for every
    variable, ``isave(1:44)`` followed by ``iwa(1:3n)`` (``move%ibfgs``),
    ``wa`` and ``dsave(1:29)``.  Numbers use gfortran's list-directed layout
    (17 significant digits), so the text is read back without loss.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (state.n,) or upper.shape != (state.n,):
        raise ValueError(f"lower and upper must have shape ({state.n},)")
    lines = [
        list_directed(state.m, float(stepmax)),
        state.task.ljust(_TASK_LENGTH),
        state.csave.ljust(_TASK_LENGTH),
    ]
    lines.extend(list_directed(bool(value)) for value in state.lsave.tolist())
    for low, high in zip(lower.tolist(), upper.tolist()):
        lines.append(list_directed(low))
        lines.append(list_directed(high))
    lines.extend(list_directed(value) for value in state.isave.tolist())
    lines.extend(list_directed(value) for value in state.iwa.tolist())
    lines.extend(list_directed(value) for value in state.wa.tolist())
    lines.extend(list_directed(value) for value in state.dsave.tolist())
    return "\n".join(lines) + "\n"


def _parse_real(text: str) -> float:
    return float(text.strip().replace("D", "E").replace("d", "e"))


def _parse_logical(text: str) -> bool:
    token = text.strip().lstrip(".").upper()
    if token.startswith("T"):
        return True
    if token.startswith("F"):
        return False
    raise ValueError(f"not a Fortran logical: {text!r}")


def parse_restart_block(
    text: str,
    n: int,
) -> tuple[LBFGSBState, np.ndarray, np.ndarray, float | None]:
    """Read the L-BFGS part of ``relax_restart.dat`` (see ``format_restart_block``).

    ``text`` must start with the record ``mmax, stepmax``, that is after the
    ``natom + 4`` records that ``domove.f90`` skips; ``n`` is three times the
    number of movable atoms.  Returns ``(state, lower, upper, stepmax)``;
    ``stepmax`` is ``None`` when the first record holds only ``mmax`` (the
    Fortran reads only ``mmax`` from it).
    """
    n = int(n)
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty restart block")
    first = lines[0].replace(",", " ").split()
    m = int(first[0])
    stepmax = _parse_real(first[1]) if len(first) > 1 else None
    nwa = _work_size(n, m)
    needed = 3 + 4 + 2 * n + 44 + 3 * n + nwa + 29
    if len(lines) < needed:
        raise ValueError(
            f"restart block for n={n}, m={m} needs {needed} records, got {len(lines)}"
        )
    task = lines[1]
    csave = lines[2]
    position = 3
    lsave = [_parse_logical(lines[position + j]) for j in range(4)]
    position += 4
    lower = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    for j in range(n):
        lower[j] = _parse_real(lines[position + 2 * j])
        upper[j] = _parse_real(lines[position + 2 * j + 1])
    position += 2 * n
    ibfgs = [int(lines[position + j]) for j in range(44 + 3 * n)]
    position += 44 + 3 * n
    wa = [_parse_real(lines[position + j]) for j in range(nwa)]
    position += nwa
    dsave = [_parse_real(lines[position + j]) for j in range(29)]
    state = LBFGSBState(
        n=n,
        m=m,
        task=task,
        csave=csave,
        lsave=np.array(lsave, dtype=bool),
        isave=np.array(ibfgs[:44], dtype=np.int32),
        dsave=np.array(dsave, dtype=np.float64),
        wa=np.array(wa, dtype=np.float64),
        iwa=np.array(ibfgs[44:], dtype=np.int32),
    )
    return state, lower, upper, stepmax
