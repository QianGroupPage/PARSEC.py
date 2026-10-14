"""Source-parity tests for the L-BFGS-B translation (PARSEC ``lbfgsb.f``).

The reference data in ``tests/data/lbfgsb_reference.json`` come from the
original Fortran sources of PARSEC (``lbfgsb.f``, ``linpack.f``, ``timer.f``),
compiled with gfortran 13.3 without optimization and linked to the netlib
reference BLAS.  A small driver called ``setulb`` with the argument
association of ``domove.f90`` and recorded every function evaluation, every
task string and the work arrays.

The objective functions below repeat the floating-point operations of that
driver one by one, so the comparison is bit for bit: every requested
evaluation point, every function value and every task string must be
identical, and so must the complete persistent state at the end of a run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from parsec_python.Relaxation import (
    UNIT_ITERATE,
    UNIT_OUT,
    LBFGSBState,
    format_restart_block,
    list_directed,
    parse_restart_block,
    setulb,
)
from parsec_python.Relaxation import lbfgsb as lbfgsb_module

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:  # pragma: no cover - SciPy is a declared requirement
    scipy_minimize = None

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "lbfgsb_reference.json"

# dsave(6:10) hold CPU times (cpu1, cachyt, sbtime, lnscht, time1).
_TIME_SLOTS = (5, 6, 7, 8, 9)


# ---------------------------------------------------------------------------
# Objective functions: the same operations, in the same order, as the
# Fortran reference driver.
# ---------------------------------------------------------------------------


def rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Chained Rosenbrock function."""
    xs = x.tolist()
    n = len(xs)
    f = 0.0
    g = [0.0] * n
    for i in range(n - 1):
        a = xs[i] * xs[i]
        t1 = xs[i + 1] - a
        t2 = 1.0 - xs[i]
        a = 100.0 * t1
        a = a * t1
        f = f + a
        a = t2 * t2
        f = f + a
        a = 400.0 * xs[i]
        a = a * t1
        g[i] = g[i] - a
        a = 2.0 * t2
        g[i] = g[i] - a
        a = 200.0 * t1
        g[i + 1] = g[i + 1] + a
    return f, np.array(g)


def driver1(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Sample function of the L-BFGS-B distribution (``driver1.f``)."""
    xs = x.tolist()
    n = len(xs)
    g = [0.0] * n
    a = xs[0] - 1.0
    a = a * a
    f = 0.25 * a
    for i in range(1, n):
        a = xs[i - 1] * xs[i - 1]
        a = xs[i] - a
        a = a * a
        f = f + a
    f = 4.0 * f
    a = xs[0] * xs[0]
    t1 = xs[1] - a
    a = xs[0] - 1.0
    a = 2.0 * a
    b = 16.0 * xs[0]
    b = b * t1
    g[0] = a - b
    for i in range(1, n - 1):
        t2 = t1
        a = xs[i] * xs[i]
        t1 = xs[i + 1] - a
        a = 8.0 * t2
        b = 16.0 * xs[i]
        b = b * t1
        g[i] = a - b
    g[n - 1] = 8.0 * t1
    return f, np.array(g)


def lennard_jones(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Lennard-Jones cluster (epsilon = sigma = 1), ``x = (x1, y1, z1, x2, ...)``."""
    xs = x.tolist()
    n = len(xs)
    natom = n // 3
    f = 0.0
    g = [0.0] * n
    for i in range(natom - 1):
        ii = 3 * i
        for j in range(i + 1, natom):
            jj = 3 * j
            dx = xs[ii] - xs[jj]
            dy = xs[ii + 1] - xs[jj + 1]
            dz = xs[ii + 2] - xs[jj + 2]
            r2 = dx * dx
            a = dy * dy
            r2 = r2 + a
            a = dz * dz
            r2 = r2 + a
            ir2 = 1.0 / r2
            ir6 = ir2 * ir2
            ir6 = ir6 * ir2
            ir12 = ir6 * ir6
            e = ir12 - ir6
            e = 4.0 * e
            f = f + e
            c = 2.0 * ir12
            c = c - ir6
            c = c * ir2
            c = -24.0 * c
            a = c * dx
            g[ii] = g[ii] + a
            g[jj] = g[jj] - a
            a = c * dy
            g[ii + 1] = g[ii + 1] + a
            g[jj + 1] = g[jj + 1] - a
            a = c * dz
            g[ii + 2] = g[ii + 2] + a
            g[jj + 2] = g[jj + 2] - a
    return f, np.array(g)


def quartic(
    x: np.ndarray, matrix: list, linear: list, fourth: list
) -> tuple[float, np.ndarray]:
    """``f = sum_i x_i (A x / 2 - b)_i + sum_i q_i x_i^4 / 4``."""
    xs = x.tolist()
    n = len(xs)
    f = 0.0
    g = [0.0] * n
    for i in range(n):
        s = 0.0
        row = matrix[i]
        for j in range(n):
            a = row[j] * xs[j]
            s = s + a
        x2 = xs[i] * xs[i]
        a = 0.5 * s
        a = a - linear[i]
        a = a * xs[i]
        f = f + a
        a = 0.25 * fourth[i]
        a = a * x2
        a = a * x2
        f = f + a
        s = s - linear[i]
        a = fourth[i] * x2
        a = a * xs[i]
        g[i] = s + a
    return f, np.array(g)


def objective_of(case: dict):
    """Return the Python objective of a fixture case."""
    problem = case["problem"]
    if problem == "rosenbrock":
        return rosenbrock
    if problem == "driver1":
        return driver1
    if problem == "lj":
        return lennard_jones
    if problem == "quartic":
        data = case["data"]
        return lambda x: quartic(x, data["A"], data["b"], data["q"])
    raise ValueError(problem)


# ---------------------------------------------------------------------------
# The reverse-communication loop of domove.f90
# ---------------------------------------------------------------------------


class Report:
    """Collects the text of Fortran units 8 (iterate.dat) and 9 (out.*)."""

    def __init__(self) -> None:
        self.parts: dict[int, list[str]] = {UNIT_ITERATE: [], UNIT_OUT: []}

    def __call__(self, unit: int, text: str) -> None:
        self.parts[unit].append(text)

    def text(self, unit: int) -> str:
        return "".join(self.parts[unit])


def drive(
    objective,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nbd: np.ndarray,
    m: int,
    factr: float,
    pgtol: float,
    *,
    state: LBFGSBState | None = None,
    stop_before_evaluation: int | None = None,
    first_evaluation: int = 1,
    iprint: int = -1,
    output=None,
    clock=None,
) -> dict:
    """Minimize as ``domove.f90`` does and record what the optimizer asks for.

    The first call has ``task = 'START'``.  Whenever the task starts with
    ``FG`` the objective is evaluated at ``x``; ``setulb`` is then called
    until it asks for another evaluation or reports a final task.  With
    ``state`` the run continues a restart state that is waiting for an
    evaluation, as PARSEC does after reading ``relax_restart.dat``.
    """
    n = x.shape[0]
    g = np.zeros(n)
    f = 0.0
    tasks: list[str] = []
    points: list[np.ndarray] = []
    values: list[float] = []
    if state is None:
        state = LBFGSBState.start(n, m)
        f = setulb(
            state, x, lower, upper, nbd, f, g, factr, pgtol, iprint=iprint,
            output=output, clock=clock,
        )
        tasks.append(state.task)
    evaluation = first_evaluation - 1
    while state.task[:2] == "FG":
        evaluation += 1
        if stop_before_evaluation is not None and evaluation == stop_before_evaluation:
            break
        f, gradient = objective(x)
        g[:] = gradient
        points.append(x.copy())
        values.append(f)
        while True:
            f = setulb(
                state, x, lower, upper, nbd, f, g, factr, pgtol, iprint=iprint,
                output=output, clock=clock,
            )
            tasks.append(state.task)
            if state.task[:5] != "NEW_X":
                break
    return {
        "state": state,
        "x": x,
        "f": f,
        "g": g,
        "tasks": tasks,
        "points": points,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Fixture access
# ---------------------------------------------------------------------------


def decode(text: str) -> np.ndarray:
    """Little-endian IEEE doubles stored as base64."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").astype(np.float64)


def encode(values) -> str:
    """Inverse of ``decode``; the fixture was written with it."""
    return base64.b64encode(np.asarray(values, dtype="<f8").tobytes()).decode("ascii")


def canonical(values) -> np.ndarray:
    """Copy as float64 with every NaN mapped to one bit pattern.

    Sign and payload of a NaN carry no information and differ between the
    Fortran program and Python.
    """
    values = np.array(values, dtype=np.float64, ndmin=1)
    values[np.isnan(values)] = np.nan
    return values


def digest(values) -> str:
    return hashlib.sha256(canonical(values).astype("<f8").tobytes()).hexdigest()


def same_bits(a, b) -> bool:
    a = canonical(a)
    b = canonical(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def largest_relative_difference(a, b) -> float:
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny)
    with np.errstate(invalid="ignore"):
        return float(np.max(np.abs(a - b) / scale))


def without_times(text: str) -> str:
    """Drop the lines of a report that hold measured CPU times."""
    return "\n".join(
        line.rstrip() for line in text.splitlines() if "seconds." not in line
    )


def defined_fields(text: str) -> str:
    """Replace what the Fortran prints for an undefined variable.

    ``prn3lb`` prints the CHARACTER*3 variable ``word`` in the line of
    ``iterate.dat`` that reports an abnormal termination, but ``mainlb`` does
    not keep ``word`` between calls: the gfortran build writes three NUL
    characters.  The Python code prints the status of the iteration, which is
    always ``---`` there.
    """
    return text.replace("\x00\x00\x00", "---")


_ABNORMAL_LINE_END = "      -          -"


def without_undefined_step(text: str) -> str:
    """Blank the ``tstep`` column of the abnormal-termination line.

    ``prn3lb`` prints ``xstep`` there, which the Fortran leaves undefined when
    the line search returns at once because the direction is not a descent
    direction (``info = -4``).
    """
    lines = []
    for line in text.split("\n"):
        if line.endswith(_ABNORMAL_LINE_END):
            line = line[:44] + "*" * 8 + line[52:]
        lines.append(line)
    return "\n".join(lines)


def load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="ascii"))


def replay_script(case: dict, iprint: int = -1, output=None) -> dict:
    """Repeat a recorded sequence of calls with the recorded ``f`` and ``g``.

    ``case["script"]`` has one letter per event of the Fortran run: ``C`` is a
    call of ``setulb``, ``E`` an evaluation (the recorded function value and
    gradient are supplied) and ``B`` the moment the caller replaces the bounds
    by ``lower2``/``upper2``.  ``injected_tasks`` are task strings that the
    caller sets before a call (``STOP`` requests).  The objective function is
    not needed, so these cases can use any function, including inconsistent
    or non-finite values.
    """
    n = case["n"]
    x, lower, upper, nbd = case_arrays(case)
    count = case["evaluation_count"]
    values = decode("".join(case["evaluation_values_b64"]))
    gradients = decode("".join(case["evaluation_gradients_b64"])).reshape(count, n)
    injected = {int(k): v for k, v in case.get("injected_tasks", {}).items()}
    state = LBFGSBState.start(n, case["m"])
    g = np.zeros(n)
    f = 0.0
    tasks: list[str] = []
    points: list[np.ndarray] = []
    evaluation = 0
    call = 0
    for event in case["script"]:
        if event == "E":
            points.append(x.copy())
            f = float(values[evaluation])
            g[:] = gradients[evaluation]
            evaluation += 1
        elif event == "B":
            lower = np.array(case["lower2"], dtype=np.float64)
            upper = np.array(case["upper2"], dtype=np.float64)
        else:
            call += 1
            if call in injected:
                state.task = injected[call]
            f = setulb(
                state, x, lower, upper, nbd, f, g, case["factr"], case["pgtol"],
                iprint=iprint, output=output,
            )
            tasks.append(state.task)
    return {
        "state": state, "x": x, "f": f, "g": g, "tasks": tasks, "points": points,
        "values": values[:evaluation].tolist(),
    }


def case_arrays(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(case["x0"], dtype=np.float64),
        np.array(case["lower"], dtype=np.float64),
        np.array(case["upper"], dtype=np.float64),
        np.array(case["nbd"], dtype=np.int64),
    )


def run_case(case: dict, objective=None, **options) -> dict:
    """Run one fixture case from ``task = 'START'``."""
    x, lower, upper, nbd = case_arrays(case)
    return drive(
        objective or objective_of(case),
        x,
        lower,
        upper,
        nbd,
        case["m"],
        case["factr"],
        case["pgtol"],
        **options,
    )


def drive_with_stop(case: dict, stop_task: str, iteration: int) -> dict:
    """Ask the optimizer to stop once ``NEW_X`` is returned for ``iteration``."""
    objective = objective_of(case)
    x, lower, upper, nbd = case_arrays(case)
    n = x.shape[0]
    g = np.zeros(n)
    f = 0.0
    arguments = (lower, upper, nbd)
    tolerances = (case["factr"], case["pgtol"])
    state = LBFGSBState.start(n, case["m"])
    tasks: list[str] = []
    points: list[np.ndarray] = []
    values: list[float] = []
    f = setulb(state, x, *arguments, f, g, *tolerances)
    tasks.append(state.task)
    stopped = False
    while state.task[:2] == "FG" and not stopped:
        f, gradient = objective(x)
        g[:] = gradient
        points.append(x.copy())
        values.append(f)
        while True:
            f = setulb(state, x, *arguments, f, g, *tolerances)
            tasks.append(state.task)
            if state.task[:5] != "NEW_X":
                break
            if state.iteration >= iteration:
                state.task = stop_task
                f = setulb(state, x, *arguments, f, g, *tolerances)
                tasks.append(state.task)
                stopped = True
                break
    return {
        "state": state, "x": x, "f": f, "g": g, "tasks": tasks, "points": points,
        "values": values,
    }


def drive_with_infinite_value(case: dict, evaluation: int) -> dict:
    """Return ``f = +inf`` (and the true gradient) at one evaluation."""
    objective = objective_of(case)
    count = 0

    def faulty(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal count
        count += 1
        f, g = objective(x)
        if count == evaluation:
            f = math.inf
        return f, g

    return run_case(case, objective=faulty)


def restart_file_header(x: np.ndarray, iteration: int) -> str:
    """The records of ``relax_restart.dat`` that precede the L-BFGS block."""
    natom = x.shape[0] // 3
    lines = [list_directed(" Atomic coordinates follow (all atoms)")]
    for j in range(natom):
        lines.append(list_directed(*x[3 * j : 3 * j + 3].tolist()))
    lines.append(list_directed("Iteration number, number of movable atoms"))
    lines.append(list_directed(iteration, natom))
    lines.append(list_directed(" LBFGS work variables follow"))
    return "\n".join(lines) + "\n"


def python_restart_file(case: dict, before_evaluation: int) -> str:
    """Run to the request for evaluation ``before_evaluation`` and write the
    restart file as ``domove.f90`` would at that point."""
    result = run_case(case, stop_before_evaluation=before_evaluation)
    _, lower, upper, _ = case_arrays(case)
    return restart_file_header(result["x"], before_evaluation - 1) + format_restart_block(
        result["state"], lower, upper, case["stepmax"]
    )


def read_restart_file(
    text: str, n: int
) -> tuple[np.ndarray, int, LBFGSBState, np.ndarray, np.ndarray]:
    """Read a restart file the way ``domove.f90`` does.

    Returns ``(x, iteration, state, lower, upper)``.
    """
    lines = text.split("\n")
    natom = n // 3
    x = np.array(
        [float(v) for line in lines[1 : 1 + natom] for v in line.split()],
        dtype=np.float64,
    )
    iteration = int(lines[natom + 2].split()[0])
    state, lower, upper, _ = parse_restart_block("\n".join(lines[natom + 4 :]), n)
    return x, iteration, state, lower, upper


def continue_from_restart_file(case: dict, text: str) -> dict:
    """Continue a run from a restart file that waits for an evaluation."""
    x, iteration, state, lower, upper = read_restart_file(text, case["n"])
    nbd = np.array(case["nbd"], dtype=np.int64)
    return drive(
        objective_of(case), x, lower, upper, nbd, case["m"], case["factr"],
        case["pgtol"], state=state, first_evaluation=iteration + 1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class _FixtureTestCase(unittest.TestCase):
    fixture: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()

    def assert_same_bits(self, actual, expected, label: str) -> None:
        if not same_bits(actual, expected):
            actual = np.atleast_1d(np.asarray(actual, dtype=np.float64))
            expected = np.atleast_1d(np.asarray(expected, dtype=np.float64))
            if actual.shape != expected.shape:
                self.fail(f"{label}: shape {actual.shape} instead of {expected.shape}")
            self.fail(
                f"{label}: not bit-identical to the Fortran reference, largest "
                f"relative difference {largest_relative_difference(actual, expected):.3e}"
            )

    def assert_same_run(self, result: dict, reference: dict, n: int, label: str) -> None:
        """Compare tasks, evaluation points and the final state with Fortran."""
        self.assertEqual(result["tasks"], reference["tasks"], f"{label}: task strings")
        count = reference["evaluation_count"]
        self.assertEqual(len(result["points"]), count, f"{label}: number of evaluations")
        if "evaluation_points_b64" in reference:
            points = decode("".join(reference["evaluation_points_b64"])).reshape(count, n)
            for number in range(count):
                self.assert_same_bits(
                    result["points"][number],
                    points[number],
                    f"{label}: point of evaluation {number + 1}",
                )
        if "evaluation_values_b64" in reference:
            values = decode("".join(reference["evaluation_values_b64"]))
            self.assert_same_bits(
                np.array(result["values"]), values, f"{label}: function values"
            )
        self.assert_same_final_state(result, reference["final"], label)

    def assert_same_final_state(self, result: dict, final: dict, label: str) -> None:
        state = result["state"]
        self.assertEqual(state.task, final["task"], f"{label}: final task")
        self.assertEqual(state.csave, final["csave"], f"{label}: final csave")
        if "f_b64" in final:
            expected_f = decode("".join(final["f_b64"]))
            expected_x = decode("".join(final["x_b64"]))
        else:
            expected_f = final["f"]
            expected_x = np.array(final["x"])
        self.assert_same_bits(result["f"], expected_f, f"{label}: final f")
        self.assert_same_bits(result["x"], expected_x, f"{label}: final x")
        self.assert_same_bits(
            result["g"], decode("".join(final["g_b64"])), f"{label}: final g"
        )
        self.assertEqual(state.lsave.tolist(), final["lsave"], f"{label}: lsave")
        self.assertEqual(state.isave.tolist(), final["isave"], f"{label}: isave")
        self.assertEqual(state.iwa.tolist(), final["iwa"], f"{label}: iwa")
        dsave = state.dsave.copy()
        dsave[list(_TIME_SLOTS)] = 0.0
        self.assert_same_bits(
            dsave, decode("".join(final["dsave_without_times_b64"])), f"{label}: dsave"
        )
        self.assertEqual(digest(state.wa), final["wa_sha256"], f"{label}: wa")


class FortranParityTests(_FixtureTestCase):
    """Complete runs against the original Fortran, bit for bit."""

    def test_every_evaluation_point_task_and_final_state(self) -> None:
        for name, case in self.fixture["cases"].items():
            with self.subTest(case=name):
                result = run_case(case)
                self.assert_same_run(result, case, case["n"], name)

    def test_natural_stops_cover_the_final_tasks_domove_tests_for(self) -> None:
        finals = {case["final"]["task"] for case in self.fixture["cases"].values()}
        self.assertEqual(
            finals,
            {
                "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
                "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL",
                "ABNORMAL_TERMINATION_IN_LNSRCH",
                "ERROR: INVALID NBD",
                "ERROR: NO FEASIBLE SOLUTION",
                "ERROR: FACTR .LT. 0",
            },
        )

    def test_parsec_bounds_become_active(self) -> None:
        # isave(39): number of active constraints at the last Cauchy point.
        for name in ("lj10_parsec_box_m7", "lj8_parsec_box_m7"):
            case = self.fixture["cases"][name]
            self.assertEqual(set(case["nbd"]), {2})
            self.assertGreater(case["final"]["isave"][38], 0)
            x = np.array(case["final"]["x"])
            lower = np.array(case["lower"])
            upper = np.array(case["upper"])
            self.assertGreater(int(np.sum((x == lower) | (x == upper))), 0)

    def test_reports_match_iterate_dat_and_unit_9(self) -> None:
        for name, case in self.fixture["cases"].items():
            with self.subTest(case=name):
                report = Report()
                run_case(case, iprint=case["iprint"], output=report)
                self.assertEqual(
                    without_times(report.text(UNIT_ITERATE)),
                    without_times(defined_fields("\n".join(case["iterate_dat"]))),
                )
                self.assertEqual(
                    without_times(report.text(UNIT_OUT)),
                    without_times(defined_fields("\n".join(case["out9"]))),
                )

    def test_verbose_reports_match(self) -> None:
        verbose = self.fixture["verbose_report"]
        case = self.fixture["cases"][verbose["case"]]
        report = Report()
        result = run_case(case, iprint=verbose["iprint"], output=report)
        self.assertEqual(result["state"].task, case["final"]["task"])
        self.assertEqual(
            without_times(report.text(UNIT_ITERATE)),
            without_times("\n".join(verbose["iterate_dat"])),
        )
        self.assertEqual(
            without_times(report.text(UNIT_OUT)),
            without_times("\n".join(verbose["out9"])),
        )

    def test_reporting_does_not_change_the_numbers(self) -> None:
        case = self.fixture["cases"]["rosenbrock_mixed_bounds_m5"]
        silent = run_case(case)
        ticks = iter(range(1, 1000000))
        verbose = run_case(
            case, iprint=101, output=Report(), clock=lambda: 0.25 * next(ticks)
        )
        self.assertEqual(silent["tasks"], verbose["tasks"])
        self.assert_same_bits(silent["x"], verbose["x"], "x")
        self.assert_same_bits(silent["state"].wa, verbose["state"].wa, "wa")
        self.assertEqual(
            silent["state"].isave.tolist(), verbose["state"].isave.tolist()
        )
        # Only the timing slots differ: they are zero without a clock.
        differing = np.nonzero(silent["state"].dsave != verbose["state"].dsave)[0]
        self.assertTrue(set(differing.tolist()) <= set(_TIME_SLOTS))
        self.assertTrue(np.all(silent["state"].dsave[list(_TIME_SLOTS)] == 0.0))
        self.assertGreater(verbose["state"].dsave[9], 0.0)

    def test_stop_requested_by_the_caller(self) -> None:
        stop = self.fixture["stop"]
        case = self.fixture["cases"][stop["case"]]
        for task, reference in stop["runs"].items():
            with self.subTest(task=task):
                result = drive_with_stop(case, task, stop["at_iteration"])
                self.assert_same_run(result, reference, case["n"], task)
        cpu = stop["runs"]["STOP: CPU EXCEEDING THE TIME LIMIT"]["final"]
        other = stop["runs"]["STOP: THE USER WANTS TO QUIT"]["final"]
        # 'STOP: CPU' restores the previous iterate, any other STOP does not.
        self.assertNotEqual(cpu["x"], other["x"])

    def test_infinite_function_value_is_handled_as_by_the_fortran(self) -> None:
        reference = self.fixture["infinite_function_value"]
        case = self.fixture["cases"][reference["case"]]
        result = drive_with_infinite_value(case, reference["at_evaluation"])
        self.assert_same_run(result, reference, case["n"], "infinite f")
        self.assertTrue(np.all(np.isfinite(result["x"])))


class RestartFileTests(_FixtureTestCase):
    """``relax_restart.dat`` written by one code and continued by the other."""

    def setUp(self) -> None:
        self.restart = self.fixture["restart"]
        self.case = self.fixture["cases"][self.restart["case"]]
        self.before = self.restart["before_evaluation"]
        self.fortran_dump = "\n".join(self.restart["fortran_dump"])

    def remaining_points(self) -> np.ndarray:
        count = self.case["evaluation_count"]
        points = decode("".join(self.case["evaluation_points_b64"]))
        return points.reshape(count, self.case["n"])[self.before - 1 :]

    def test_fortran_state_is_continued_by_python(self) -> None:
        x, iteration, state, _, _ = read_restart_file(self.fortran_dump, self.case["n"])
        self.assertEqual(iteration, self.before - 1)
        self.assertEqual(state.task, "FG_LNSRCH")
        self.assertGreater(state.isave[30], state.m)  # the memory has wrapped
        result = continue_from_restart_file(self.case, self.fortran_dump)
        expected = self.remaining_points()
        self.assertEqual(len(result["points"]), expected.shape[0])
        self.assert_same_bits(x, expected[0], "coordinates in the restart file")
        for number, point in enumerate(result["points"]):
            self.assert_same_bits(
                point, expected[number], f"point of evaluation {self.before + number}"
            )
        calls = len(result["tasks"])
        self.assertEqual(result["tasks"], self.case["tasks"][-calls:])
        # The complete final state equals that of the uninterrupted Fortran run.
        self.assert_same_final_state(result, self.case["final"], "continued run")

    def test_python_state_is_continued_by_fortran(self) -> None:
        text = python_restart_file(self.case, self.before)
        self.assertEqual(
            hashlib.sha256(text.encode("ascii")).hexdigest(),
            self.restart["python_dump_sha256"],
            "the Python restart file is not the one the Fortran continued from",
        )
        continued = self.restart["fortran_continuation_of_python_dump"]
        # What the Fortran did with the Python file is what it did without
        # the interruption.
        count = continued["evaluation_count"]
        points = decode("".join(continued["evaluation_points_b64"]))
        self.assert_same_bits(
            points.reshape(count, self.case["n"]),
            self.remaining_points(),
            "evaluation points of the Fortran continuation",
        )
        values = decode("".join(self.case["evaluation_values_b64"]))
        self.assert_same_bits(
            decode("".join(continued["evaluation_values_b64"])),
            values[self.before - 1 :],
            "function values of the Fortran continuation",
        )
        calls = len(continued["tasks"])
        self.assertEqual(continued["tasks"], self.case["tasks"][-calls:])
        for key in ("task", "f", "x", "g_b64", "lsave", "isave", "iwa", "wa_sha256"):
            self.assertEqual(continued["final"][key], self.case["final"][key], key)

    def test_python_file_equals_fortran_file_except_cpu_times(self) -> None:
        python_lines = python_restart_file(self.case, self.before).split("\n")
        fortran_lines = self.restart["fortran_dump"]
        self.assertEqual(len(python_lines), len(fortran_lines))
        differing = [
            number
            for number, (a, b) in enumerate(zip(python_lines, fortran_lines))
            if a != b
        ]
        first_dsave = len(fortran_lines) - 1 - 29
        self.assertEqual(differing, [first_dsave + slot for slot in _TIME_SLOTS])

    def test_restart_block_is_formatted_as_gfortran_writes_it(self) -> None:
        n = self.case["n"]
        lines = self.restart["fortran_dump"]
        block = "\n".join(lines[n // 3 + 4 :])
        state, lower, upper, stepmax = parse_restart_block(block, n)
        self.assertEqual(state.m, self.case["m"])
        self.assertEqual(stepmax, self.case["stepmax"])
        self.assert_same_bits(lower, np.array(self.case["lower"]), "lower")
        self.assert_same_bits(upper, np.array(self.case["upper"]), "upper")
        self.assertEqual(format_restart_block(state, lower, upper, stepmax), block)
        header = restart_file_header(
            read_restart_file(self.fortran_dump, n)[0], self.before - 1
        )
        self.assertEqual(header, "\n".join(lines[: n // 3 + 4]) + "\n")

    def test_file_of_a_parsec_run_is_reproduced_byte_for_byte(self) -> None:
        # relax_restart.dat of a BFGS relaxation of a water molecule by the
        # Fortran PARSEC (9 variables, m = 7, no bounds), stored as numbers;
        # writing them must give back the file PARSEC wrote.
        reference = self.fixture["parsec_water_restart"]
        state = LBFGSBState(
            n=reference["n"],
            m=reference["m"],
            task=reference["task"],
            csave=reference["csave"],
            lsave=np.array(reference["lsave"], dtype=bool),
            isave=np.array(reference["isave"], dtype=np.int32),
            dsave=decode("".join(reference["dsave_b64"])),
            wa=decode("".join(reference["wa_b64"])),
            iwa=np.array(reference["iwa"], dtype=np.int32),
        )
        self.assertEqual(state.task, "FG_LNSRCH")
        self.assertEqual(state.iteration, 4)
        self.assertEqual(state.function_evaluations, 7)
        text = restart_file_header(
            np.array(reference["coordinates"]), reference["iteration"]
        ) + format_restart_block(
            state,
            np.array(reference["lower"]),
            np.array(reference["upper"]),
            reference["stepmax"],
        )
        self.assertEqual(len(text), reference["file_bytes"])
        self.assertEqual(
            hashlib.sha256(text.encode("ascii")).hexdigest(), reference["file_sha256"]
        )
        parsed, lower, upper, stepmax = parse_restart_block(
            text.split("\n", reference["n"] // 3 + 4)[-1], reference["n"]
        )
        self.assertEqual(stepmax, reference["stepmax"])
        self.assertEqual(parsed.isave.tolist(), reference["isave"])
        self.assertEqual(parsed.wa.tobytes(), state.wa.tobytes())
        self.assertEqual(parsed.dsave.tobytes(), state.dsave.tobytes())

    def test_restart_block_errors(self) -> None:
        n = self.case["n"]
        block = "\n".join(self.restart["fortran_dump"][n // 3 + 4 :])
        with self.assertRaises(ValueError):
            parse_restart_block(block, n + 3)
        with self.assertRaises(ValueError):
            parse_restart_block("", n)
        state, lower, upper, _ = parse_restart_block(block, n)
        with self.assertRaises(ValueError):
            format_restart_block(state, lower[:-1], upper, 0.1)


class RarelyTakenBranchTests(_FixtureTestCase):
    """Short runs picked from a randomized comparison with the Fortran so that
    the branches which the regular cases do not reach are executed: failed
    Cholesky factorizations with a refresh of the memory, ascent directions,
    exactly zero gradient components, stationary starts, bounds replaced by
    the caller, STOP requests, calls after a final task, NaN and infinity."""

    def test_recorded_runs_are_reproduced(self) -> None:
        for case in self.fixture["branch_cases"]["cases"]:
            with self.subTest(case=case["name"]):
                report = Report()
                result = replay_script(case, iprint=case["iprint"], output=report)
                self.assert_same_run(result, case, case["n"], case["name"])
                for unit, key in ((UNIT_ITERATE, "iterate_dat"), (UNIT_OUT, "out9")):
                    if key not in case:
                        continue
                    self.assertEqual(
                        without_undefined_step(without_times(report.text(unit))),
                        without_undefined_step(
                            without_times(defined_fields("\n".join(case[key])))
                        ),
                        key,
                    )

    def test_selection_reaches_the_failure_branches(self) -> None:
        cases = self.fixture["branch_cases"]["cases"]
        reports = "\n".join("\n".join(case.get("out9", [])) for case in cases)
        for message in self.fixture["branch_cases"]["messages"]:
            self.assertIn(message, reports)
        finals = {case["final"]["task"] for case in cases}
        self.assertIn("ABNORMAL_TERMINATION_IN_LNSRCH", finals)


class StateAndInterfaceTests(unittest.TestCase):
    def test_start_state_has_the_fortran_array_sizes(self) -> None:
        state = LBFGSBState.start(9, 7)
        self.assertEqual(state.task, "START")
        self.assertEqual(state.csave, "")
        self.assertEqual(state.wa.shape, (2 * 7 * 9 + 5 * 9 + 11 * 49 + 8 * 7,))
        self.assertEqual(state.iwa.shape, (27,))
        self.assertEqual(state.isave.shape, (44,))
        self.assertEqual(state.dsave.shape, (29,))
        self.assertEqual(state.lsave.shape, (4,))
        with self.assertRaises(ValueError):
            LBFGSBState.start(0, 7)
        with self.assertRaises(ValueError):
            LBFGSBState.start(3, 0)

    def test_first_call_partitions_the_work_array_as_setulb_does(self) -> None:
        # isave(1:16) of the PARSEC water relaxation (n = 9, m = 7), read
        # from its relax_restart.dat.
        n, m = 9, 7
        state = LBFGSBState.start(n, m)
        x = np.linspace(-0.4, 0.4, n)
        g = np.zeros(n)
        zeros = np.zeros(n)
        f = setulb(state, x, zeros, zeros, np.zeros(n, dtype=int), 0.0, g, 1e-15, 1e-15)
        self.assertEqual(f, 0.0)
        self.assertEqual(state.task, "FG_START")
        self.assertEqual(
            state.isave[:16].tolist(),
            [63, 49, 196, 1, 64, 127, 176, 225, 274, 470, 666, 675, 684, 693, 702, 711],
        )
        self.assertEqual(state.isave[23], UNIT_ITERATE)
        self.assertEqual(state.iwa[n : 2 * n].tolist(), [-1] * n)
        self.assertEqual(state.dsave[4], np.finfo(float).eps)
        self.assertEqual(state.dsave[2], 1e-15 * np.finfo(float).eps)

    def test_copy_is_independent(self) -> None:
        state = LBFGSBState.start(3, 2)
        clone = state.copy()
        clone.wa[0] = 1.0
        clone.isave[0] = 5
        clone.task = "NEW_X"
        self.assertEqual(state.wa[0], 0.0)
        self.assertEqual(state.isave[0], 0)
        self.assertEqual(state.task, "START")

    def test_state_properties(self) -> None:
        x = np.array([-1.2, 1.0, -1.2, 1.0])
        zeros = np.zeros(4)
        result = drive(rosenbrock, x, zeros, zeros, np.zeros(4, dtype=int), 5, 1e7, 1e-5)
        state = result["state"]
        self.assertEqual(state.iteration, int(state.isave[29]))
        self.assertEqual(state.function_evaluations, len(result["points"]))
        self.assertEqual(state.projected_gradient_norm, float(state.dsave[12]))
        self.assertLessEqual(state.projected_gradient_norm, 1e-3)
        self.assertLess(np.max(np.abs(x - 1.0)), 1e-4)

    def test_argument_checks(self) -> None:
        n = 3
        state = LBFGSBState.start(n, 2)
        x = np.zeros(n)
        g = np.zeros(n)
        zeros = np.zeros(n)
        nbd = np.zeros(n, dtype=int)
        with self.assertRaises(ValueError):
            setulb(state, np.zeros(n + 1), zeros, zeros, nbd, 0.0, g, 1.0, 1.0)
        with self.assertRaises(ValueError):
            setulb(state, x, zeros, zeros, nbd, 0.0, g.astype(np.float32), 1.0, 1.0)
        with self.assertRaises(ValueError):
            setulb(state, x, zeros[:-1], zeros, nbd, 0.0, g, 1.0, 1.0)
        with self.assertRaises(ValueError):
            setulb(state, x, zeros, zeros, nbd.astype(float), 0.0, g, 1.0, 1.0)
        frozen = np.zeros(n)
        frozen.setflags(write=False)
        with self.assertRaises(ValueError):
            setulb(state, frozen, zeros, zeros, nbd, 0.0, g, 1.0, 1.0)
        # A state that was never started has no partition of wa.
        state.task = "FG_LNSRCH"
        with self.assertRaises(ValueError):
            setulb(state, x, zeros, zeros, nbd, 0.0, g, 1.0, 1.0)
        with self.assertRaises(ValueError):
            LBFGSBState(
                n=n, m=2, task="START", csave="", lsave=np.zeros(4, dtype=bool),
                isave=np.zeros(44, dtype=np.int32), dsave=np.zeros(29),
                wa=np.zeros(5), iwa=np.zeros(3 * n, dtype=np.int32),
            )

    def test_task_strings_drop_the_fortran_blank_padding(self) -> None:
        state = LBFGSBState.start(2, 2)
        state = LBFGSBState(
            n=2, m=2, task="START".ljust(60), csave=" ".ljust(60), lsave=state.lsave,
            isave=state.isave, dsave=state.dsave, wa=state.wa, iwa=state.iwa,
        )
        self.assertEqual(state.task, "START")
        self.assertEqual(state.csave, "")

    def test_call_after_a_final_task_asks_for_f_and_g_again(self) -> None:
        # mainlb falls through to 'FG_START' for a task it does not know.
        x = np.array([-1.2, 1.0])
        zeros = np.zeros(2)
        nbd = np.zeros(2, dtype=int)
        result = drive(rosenbrock, x, zeros, zeros, nbd, 5, 1e7, 1e-5)
        state = result["state"]
        self.assertEqual(state.task[:4], "CONV")
        setulb(state, x, zeros, zeros, nbd, result["f"], result["g"], 1e7, 1e-5)
        self.assertEqual(state.task, "FG_START")


def from_hex(text: str) -> float:
    """IEEE double from its hexadecimal image (Fortran ``Z16.16``)."""
    return float(np.frombuffer(bytes.fromhex(text), dtype=">f8")[0])


def to_hex(value: float) -> str:
    return np.array([value], dtype=">f8").tobytes().hex().upper()


def quartic_polynomial(c: list, stp: float) -> tuple[float, float]:
    """Value and derivative of ``c0 + c1 s + ... + c4 s^4`` at ``stp``, with
    the operations of the Fortran probe."""
    f = c[4]
    f = f * stp
    f = f + c[3]
    f = f * stp
    f = f + c[2]
    f = f * stp
    f = f + c[1]
    f = f * stp
    f = f + c[0]
    g = 4.0 * c[4]
    g = g * stp
    t = 3.0 * c[3]
    g = g + t
    g = g * stp
    t = 2.0 * c[2]
    g = g + t
    g = g * stp
    g = g + c[1]
    return f, g


class LineSearchReferenceTests(_FixtureTestCase):
    """``dcstep`` and ``dcsrch`` called directly, against the Fortran routines."""

    def test_dcstep_matches_fortran_in_every_branch(self) -> None:
        samples = self.fixture["line_search"]["dcstep"]
        # Two function-value/derivative cases with stp on either side of stx,
        # two more that also depend on whether a minimizer is bracketed.
        self.assertEqual(len(samples), 12)
        for branch, records in samples.items():
            for record in records:
                with self.subTest(branch=branch):
                    values = [from_hex(v) for v in record["input"]]
                    result = lbfgsb_module._dcstep(
                        *values[:9], record["brackt"], *values[9:]
                    )
                    self.assertEqual(
                        [to_hex(v) for v in result[:7]], record["output"]
                    )
                    self.assertEqual(result[7], record["brackt_out"])

    def test_dcsrch_line_searches_match_fortran(self) -> None:
        samples = self.fixture["line_search"]["dcsrch"]
        self.assertIn("CONVERGENCE", samples)
        self.assertIn("WARNING: XTOL TEST SATISFIED", samples)
        self.assertIn("WARNING: STP = STPMAX", samples)
        self.assertIn("WARNING: STP = STPMIN", samples)
        self.assertIn("ERROR: STP .GT. STPMAX", samples)
        for final_task, sample in samples.items():
            with self.subTest(final_task=final_task):
                c = [from_hex(v) for v in sample["polynomial"]]
                stp = from_hex(sample["stp"])
                stpmin = from_hex(sample["stpmin"])
                stpmax = from_hex(sample["stpmax"])
                isave = np.zeros(2, dtype=np.int32)
                dsave = np.zeros(13)
                task = "START"
                f, g = c[0], c[1]
                for call in sample["calls"]:
                    stp, task = lbfgsb_module._dcsrch(
                        f, g, stp, sample["ftol"], sample["gtol"], sample["xtol"],
                        stpmin, stpmax, task, isave, dsave,
                    )
                    self.assertEqual(task, call["task"])
                    self.assertEqual(to_hex(stp), call["stp"])
                    f, g = quartic_polynomial(c, stp)
                self.assertEqual(task, final_task)


class LinpackAndLineSearchTests(unittest.TestCase):
    def test_dpofa_factors_and_reports_the_failing_minor(self) -> None:
        rng = np.random.default_rng(1)
        root = rng.normal(size=(6, 6))
        matrix = root.T @ root + 0.1 * np.eye(6)
        work = matrix.tolist()
        self.assertEqual(lbfgsb_module._dpofa(work, 6), 0)
        factor = np.triu(np.array(work))
        np.testing.assert_allclose(factor.T @ factor, matrix, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            factor, np.linalg.cholesky(matrix).T, rtol=1e-12, atol=1e-13
        )
        # The strict lower triangle is not touched.
        np.testing.assert_array_equal(np.tril(np.array(work), -1), np.tril(matrix, -1))
        # A trailing block, as formk uses for the (2,2) block of wn.
        block = np.zeros((8, 8))
        block[2:, 2:] = matrix
        work = block.tolist()
        self.assertEqual(lbfgsb_module._dpofa(work, 6, 2), 0)
        np.testing.assert_array_equal(np.triu(np.array(work)[2:, 2:]), factor)
        indefinite = np.diag([1.0, 2.0, -1.0, 4.0]).tolist()
        self.assertEqual(lbfgsb_module._dpofa(indefinite, 4), 3)

    def test_dtrsl_solves_the_four_kinds_of_systems(self) -> None:
        rng = np.random.default_rng(2)
        n = 7
        full = rng.normal(size=(n, n)) + 4.0 * np.eye(n)
        rhs = rng.normal(size=n)
        lower = np.tril(full)
        upper = np.triu(full)
        systems = {0: lower, 1: upper, 10: lower.T, 11: upper.T}
        for job, matrix in systems.items():
            with self.subTest(job=job):
                work = [0.0, 0.0] + rhs.tolist()
                info = lbfgsb_module._dtrsl(full.tolist(), n, work, 2, job)
                self.assertEqual(info, 0)
                np.testing.assert_allclose(
                    np.array(work[2:]), np.linalg.solve(matrix, rhs), rtol=1e-12
                )
                self.assertEqual(work[:2], [0.0, 0.0])
        singular = full.copy()
        singular[4, 4] = 0.0
        work = rhs.tolist()
        self.assertEqual(lbfgsb_module._dtrsl(singular.tolist(), n, work, 0, 11), 5)
        self.assertEqual(work, rhs.tolist())

    def test_ddot_is_the_sequential_sum_of_the_reference_blas(self) -> None:
        rng = np.random.default_rng(3)
        for size in (0, 1, 2, 5, 39, 1000):
            a = rng.normal(size=size) * 10.0 ** rng.integers(-8, 8, size=size)
            b = rng.normal(size=size)
            total = 0.0
            for left, right in zip(a.tolist(), b.tolist()):
                total = total + left * right
            self.assertEqual(lbfgsb_module._ddot(a, b), total)
        self.assertEqual(
            math.copysign(1.0, lbfgsb_module._ddot(np.array([-0.0]), np.array([1.0]))),
            1.0,
        )

    def test_dcsrch_reports_invalid_input_without_touching_its_state(self) -> None:
        isave = np.array([7, 7], dtype=np.int32)
        dsave = np.full(13, 3.0)
        stp, task = lbfgsb_module._dcsrch(
            1.0, -1.0, 2.0, 1e-3, 0.9, 0.1, 0.0, 1.0, "START", isave, dsave
        )
        self.assertEqual((stp, task), (2.0, "ERROR: STP .GT. STPMAX"))
        self.assertEqual(isave.tolist(), [7, 7])
        self.assertTrue(np.all(dsave == 3.0))
        _, task = lbfgsb_module._dcsrch(
            1.0, 0.5, 0.5, 1e-3, 0.9, 0.1, 0.0, 1.0, "START", isave, dsave
        )
        self.assertEqual(task, "ERROR: INITIAL G .GE. ZERO")
        _, task = lbfgsb_module._dcsrch(
            1.0, -1.0, -0.5, 1e-3, 0.9, 0.1, 0.0, 1.0, "START", isave, dsave
        )
        self.assertEqual(task, "ERROR: STP .LT. STPMIN")

    def test_dcsrch_accepts_a_step_that_satisfies_both_conditions(self) -> None:
        # phi(s) = (s - 1)^2 along the step: phi(0) = 1, phi'(0) = -2.
        isave = np.zeros(2, dtype=np.int32)
        dsave = np.zeros(13)
        stp, task = lbfgsb_module._dcsrch(
            1.0, -2.0, 1.0, 1e-3, 0.9, 0.1, 0.0, 10.0, "START", isave, dsave
        )
        self.assertEqual((stp, task), (1.0, "FG"))
        stp, task = lbfgsb_module._dcsrch(
            0.0, 0.0, stp, 1e-3, 0.9, 0.1, 0.0, 10.0, task, isave, dsave
        )
        self.assertEqual((stp, task), (1.0, "CONVERGENCE"))

    def test_division_and_square_root_follow_ieee_arithmetic(self) -> None:
        self.assertEqual(lbfgsb_module._div(1.0, 0.0), math.inf)
        self.assertEqual(lbfgsb_module._div(-1.0, 0.0), -math.inf)
        self.assertEqual(lbfgsb_module._div(1.0, -0.0), -math.inf)
        self.assertTrue(math.isnan(lbfgsb_module._div(0.0, 0.0)))
        self.assertTrue(math.isnan(lbfgsb_module._sqrt(-1.0)))
        self.assertEqual(lbfgsb_module._sqrt(4.0), 2.0)


@unittest.skipIf(scipy_minimize is None, "SciPy is not installed")
class SciPyComparisonTests(_FixtureTestCase):
    """``scipy.optimize`` implements the same L-BFGS-B 3.0 with other
    arithmetic (its own BLAS), so its evaluation points agree at rounding
    level at first and drift apart slowly afterwards."""

    def scipy_run(self, case: dict) -> dict:
        objective = objective_of(case)
        points: list[np.ndarray] = []

        def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
            points.append(np.array(x, dtype=np.float64))
            return objective(np.asarray(x, dtype=np.float64))

        bounds = [
            (low if kind in (1, 2) else None, high if kind in (2, 3) else None)
            for kind, low, high in zip(case["nbd"], case["lower"], case["upper"])
        ]
        result = scipy_minimize(
            fun,
            np.array(case["x0"], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxcor": case["m"],
                "ftol": case["factr"] * np.finfo(float).eps,
                "gtol": case["pgtol"],
                "maxls": 20,
                "maxfun": 100000,
                "maxiter": 100000,
            },
        )
        return {"points": points, "result": result}

    def test_unconstrained_runs(self) -> None:
        names = (
            "lj13_unconstrained_m7",
            "lj8_unconstrained_m3",
            "rosenbrock_unconstrained_m7",
            "rosenbrock_unconstrained_m17",
        )
        for name in names:
            case = self.fixture["cases"][name]
            with self.subTest(case=name):
                ours = run_case(case)
                theirs = self.scipy_run(case)
                for number in range(6):
                    self.assertLess(
                        largest_relative_difference(
                            ours["points"][number], theirs["points"][number]
                        ),
                        1e-10,
                        f"evaluation {number + 1}",
                    )
                # Both stop where f no longer decreases, which fixes x only to
                # about the square root of the rounding error of f.
                np.testing.assert_allclose(
                    ours["x"], theirs["result"].x, rtol=1e-4, atol=1e-5
                )
                self.assertLess(
                    abs(ours["f"] - theirs["result"].fun),
                    1e-10 * max(1.0, abs(ours["f"])),
                )

    def test_bounded_sample_problem_agrees_at_every_evaluation(self) -> None:
        case = self.fixture["cases"]["driver1_sample_m5"]
        ours = run_case(case)
        theirs = self.scipy_run(case)
        self.assertEqual(len(ours["points"]), len(theirs["points"]))
        for number, point in enumerate(ours["points"]):
            self.assertLess(
                largest_relative_difference(point, theirs["points"][number]),
                1e-10,
                f"evaluation {number + 1}",
            )


class GfortranFormatTests(_FixtureTestCase):
    def test_edit_descriptors_and_list_directed_reals(self) -> None:
        for sample in self.fixture["gfortran_formats"]["samples"]:
            value = float(np.frombuffer(bytes.fromhex(sample["bits"]), dtype=">f8")[0])
            with self.subTest(bits=sample["bits"]):
                self.assertEqual(lbfgsb_module._fd(value, 12, 5), sample["d12.5"])
                self.assertEqual(lbfgsb_module._fd(value, 10, 3), sample["d10.3"])
                self.assertEqual(lbfgsb_module._fd(value, 8, 1), sample["d8.1"])
                self.assertEqual(lbfgsb_module._fd(value, 11, 4), sample["d11.4"])
                self.assertEqual(lbfgsb_module._fe(value, 10, 3), sample["e10.3"])
                self.assertEqual(list_directed(value), sample["list"])
                self.assertEqual(
                    list_directed("x", value, "y", value, value), sample["list_mixed"]
                )

    def test_integer_fields_and_vector_blocks(self) -> None:
        # Output of gfortran for (i5) and for the format
        # (/,a4,1p,6(1x,d11.4),/,(4x,1p,6(1x,d11.4))) with 4, 6 and 8 values:
        # with exactly six values the second slash still starts a record.
        self.assertEqual(lbfgsb_module._fi(37035, 5), "37035")
        self.assertEqual(lbfgsb_module._fi(-37035, 5), "*****")
        values = [1.5 * k for k in range(1, 9)]
        first = " X =  1.5000D+00  3.0000D+00  4.5000D+00  6.0000D+00"
        self.assertEqual(
            lbfgsb_module._vector_block("X =", values[:4]), "\n" + first + "\n"
        )
        first += "  7.5000D+00  9.0000D+00"
        self.assertEqual(
            lbfgsb_module._vector_block("X =", values[:6]), "\n" + first + "\n\n"
        )
        self.assertEqual(
            lbfgsb_module._vector_block("X =", values),
            "\n" + first + "\n      1.0500D+01  1.2000D+01\n",
        )
        self.assertEqual(
            lbfgsb_module._cauchy_block(values[:2]),
            "Cauchy X =  \n      1.5000D+00  3.0000D+00\n",
        )

    def test_list_directed_records_of_relax_restart(self) -> None:
        # Records of the relax_restart.dat of a PARSEC water relaxation.
        self.assertEqual(
            list_directed(0.0, -0.67587172613767599, -5.5108084382129065e-35),
            "   0.0000000000000000      -0.67587172613767599       "
            "-5.5108084382129065E-035",
        )
        self.assertEqual(list_directed(5, 3), "           5           3")
        self.assertEqual(list_directed(7, -1.0), "           7  -1.0000000000000000     ")
        self.assertEqual(list_directed(True), " T")
        self.assertEqual(list_directed(False), " F")
        self.assertEqual(
            list_directed(" LBFGS work variables follow"),
            "  LBFGS work variables follow",
        )
        self.assertEqual(list_directed("N = ", 9, "    M = ", 7),
                         " N =            9     M =            7")
        self.assertEqual(list_directed(), "")
        self.assertEqual(list_directed(np.int32(-12), np.float64(2.5), np.bool_(True)),
                         "         -12   2.5000000000000000      T")
        with self.assertRaises(TypeError):
            list_directed(None)


if __name__ == "__main__":
    unittest.main()
