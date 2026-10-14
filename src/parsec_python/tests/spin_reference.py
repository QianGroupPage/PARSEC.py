"""Shared helpers for tests against spin-polarized runs of Fortran PARSEC.

``data/spin_parsec_reference.json`` holds, for a few small isolated systems,
the input Fortran PARSEC was run with (``Spin_Polarization: .true.``) and
what its ``parsec.out`` reports for the last SCF step: the energy terms, the
levels and occupations of each spin channel, the electrons per channel, the
Fermi level and the force tables.  The pseudopotentials are files of the test
data: ``spin_*_POTRE.DAT`` are the potentials of the library under
``src/pp_generation`` as the Fortran runs read them (written while the
converter still resampled linearly), the others belong to a core-corrected
case.

The helpers write such an input with its files into a directory and compare
a result of this package with the Fortran numbers.  Two solutions that
differ by the exchange of the spin channels have the same energy, and which
one an SCF loop reaches depends on its first steps: the comparison accepts
either and says so through :func:`channels_in_fortran_order`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np


DATA = Path(__file__).resolve().parent / "data"
#: The pseudopotential library, for cases that are not compared with Fortran.
LIBRARY = Path(__file__).resolve().parents[2] / "pp_generation"

#: Fortran prints eight decimals of an energy and ten of a level; both SCF
#: loops stop below 1e-8 Ry.
ENERGY_TOLERANCE = 3.0e-8
TERM_TOLERANCE = 2.0e-7
LEVEL_TOLERANCE = 2.0e-7
#: Four decimals of an occupation and of the Fermi level are printed.
PRINTED_FOUR_DECIMALS = 5.01e-5
#: Six decimals of a force are printed.
FORCE_TOLERANCE = 1.5e-6


def load_cases() -> dict:
    return json.loads((DATA / "spin_parsec_reference.json").read_text())["cases"]


def pseudopotential_file(source: str) -> Path:
    """Locate a pseudopotential the fixture names.

    A name that starts with ``tests/data/`` is a file of the test data; any
    other is relative to the library under ``src/pp_generation``.
    """

    prefix = "tests/data/"
    if source.startswith(prefix):
        return DATA / source[len(prefix) :]
    return LIBRARY / source


def write_case(
    case: dict,
    directory: Path,
    *,
    parsec_boundary: bool = True,
    extra: tuple[str, ...] = (),
) -> Path:
    """Write ``parsec.in`` and every file of ``case`` into ``directory``.

    ``parsec_boundary`` adds ``Hartree_Boundary_Tolerance: off``, with which
    this package uses the Hartree boundary values of PARSEC.  ``extra``
    lines are appended to the input.
    """

    directory = Path(directory)
    lines = list(case["input"])
    if parsec_boundary:
        lines.append("Hartree_Boundary_Tolerance: off")
    lines.extend(extra)
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    for target, source in case["pseudopotentials"].items():
        shutil.copyfile(pseudopotential_file(source), directory / target)
    if "occup" in case:
        (directory / "occup.in").write_text("\n".join(case["occup"]) + "\n")
    return directory / "parsec.in"


def channels_in_fortran_order(result, case: dict) -> tuple:
    """Return the two channels of ``result``, the majority one where Fortran has it.

    The channels are exchanged when the net moments of the two runs have
    opposite signs; a run without a moment keeps its order.
    """

    up, down = result.spin
    fortran_moment = case["electrons"][0] - case["electrons"][1]
    moment = up.electron_count - down.electron_count
    if fortran_moment * moment < -1.0e-6:
        return (down, up)
    return (up, down)


def assert_matches_fortran(
    test,
    result,
    case: dict,
    *,
    energy_tolerance: float = ENERGY_TOLERANCE,
    term_tolerance: float = TERM_TOLERANCE,
    level_tolerance: float = LEVEL_TOLERANCE,
) -> None:
    """Compare energies, levels, occupations and electrons with Fortran."""

    test.assertTrue(result.converged)
    test.assertAlmostEqual(result.energies.total, case["energy"], delta=energy_tolerance)
    for name, value in case["energies"].items():
        test.assertAlmostEqual(
            getattr(result.energies, name), value, delta=term_tolerance, msg=name
        )
    channels = channels_in_fortran_order(result, case)
    for label, channel, electrons in zip(("up", "down"), channels, case["electrons"]):
        np.testing.assert_allclose(
            channel.eigenvalues,
            case["eigenvalues"][label],
            rtol=0,
            atol=level_tolerance,
            err_msg=f"levels of the {label} channel",
        )
        np.testing.assert_allclose(
            channel.occupations,
            case["occupations"][label],
            rtol=0,
            atol=PRINTED_FOUR_DECIMALS,
            err_msg=f"occupations of the {label} channel",
        )
        # Two decimals of the electrons per channel are printed.
        test.assertAlmostEqual(channel.electron_count, electrons, delta=5.01e-3)
    test.assertAlmostEqual(
        abs(result.net_magnetic_moment), abs(case["net_magnetic_moment"]), delta=1.0e-2
    )
    if "fermi_level" in case:
        test.assertAlmostEqual(
            result.fermi_level, case["fermi_level"], delta=PRINTED_FOUR_DECIMALS
        )
    else:
        test.assertTrue(np.isnan(result.fermi_level))


def assert_forces_match(test, forces, case: dict, *, tolerance: float = FORCE_TOLERANCE) -> None:
    """Compare every force table with the Fortran numbers."""

    tables = {
        "positions": forces.positions,
        "ionic": forces.local.ionic,
        "sc": forces.local.self_consistency,
        "core": forces.local.core_correction,
        "local": forces.local_with_ion_ion,
        "nonlocal": forces.nonlocal_pseudopotential,
        "symmetrized": forces.symmetrized,
        "net": forces.net_force,
        "total": forces.total,
    }
    for name, values in tables.items():
        np.testing.assert_allclose(
            values,
            np.asarray(case["forces"][name]),
            rtol=0,
            atol=tolerance,
            err_msg=f"force table {name!r}",
        )
