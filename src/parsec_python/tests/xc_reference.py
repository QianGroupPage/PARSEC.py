"""Shared helpers for tests against Fortran PARSEC runs of more functionals.

``data/xc_parsec_reference.json`` holds Fortran runs of the water, O2 and
CH3 inputs of ``data/spin_parsec_reference.json`` with other
``Correlation_Type`` values (PW92, X-alpha, Wigner, Hedin-Lundqvist, LB94,
Casida-Salahub, PW91, PBE) and what ``parsec.out`` reports for the last SCF
step: the total energy, the energy terms and the levels (``all`` without
spin, ``up``/``dn`` with spin).  The pseudopotentials are the CA files of
the spin fixture, for which Fortran warns with every other functional.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .spin_reference import pseudopotential_file


DATA = Path(__file__).resolve().parent / "data"
#: Fortran prints eight decimals of an energy and ten of a level.
ENERGY_TOLERANCE = 3.0e-8
TERM_TOLERANCE = 2.0e-7
LEVEL_TOLERANCE = 2.0e-7


def load_xc_cases() -> dict:
    return json.loads((DATA / "xc_parsec_reference.json").read_text())["cases"]


def write_xc_case(case: dict, directory: Path, *, extra: tuple[str, ...] = ()) -> Path:
    """Write ``parsec.in`` (with PARSEC's Hartree boundary) and the files of ``case``."""

    directory = Path(directory)
    lines = [*case["input"], "Hartree_Boundary_Tolerance: off", *extra]
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    for target, source in case["pseudopotentials"].items():
        shutil.copyfile(pseudopotential_file(source), directory / target)
    return directory / "parsec.in"


def result_levels(result) -> dict[str, np.ndarray]:
    """The levels of ``result`` under the channel names of the fixture."""

    spin = getattr(result, "spin", None)
    if getattr(result, "spin_polarized", False) and spin is not None:
        up, down = spin
        return {"up": np.asarray(up.eigenvalues), "dn": np.asarray(down.eigenvalues)}
    return {"all": np.asarray(result.eigenvalues)}


def assert_matches_fortran_case(
    test,
    result,
    case: dict,
    *,
    energy_tolerance: float = ENERGY_TOLERANCE,
    term_tolerance: float = TERM_TOLERANCE,
    level_tolerance: float = LEVEL_TOLERANCE,
    terms: tuple[str, ...] | None = None,
) -> None:
    """Compare the total energy, the energy terms and every level with Fortran."""

    test.assertTrue(result.converged)
    test.assertAlmostEqual(result.energies.total, case["energy"], delta=energy_tolerance)
    for name, value in case["energies"].items():
        if terms is not None and name not in terms:
            continue
        test.assertAlmostEqual(
            getattr(result.energies, name), value, delta=term_tolerance, msg=name
        )
    levels = result_levels(result)
    for channel, reference in case["eigenvalues"].items():
        np.testing.assert_allclose(
            levels[channel],
            reference,
            rtol=0,
            atol=level_tolerance,
            err_msg=f"levels of channel {channel!r}",
        )
